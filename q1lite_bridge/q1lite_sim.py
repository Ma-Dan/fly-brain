"""
Q1LiteSim: Torque-controlled MuJoCo wrapper for the Q1 Lite spider quadruped.

Q1 Lite (Jason Workshop): 8-DOF, 80 g, 44 mm standing height.
    hip  = YAW about Z (swings foot fore/aft)
    knee = LIFT about Y (raises/lowers foot)
    Forward = body +Y (head side, "F" legs); right = +X; up = +Z.

Uses PD torque control matching q1lite/view_demo.py:
    ctrl[i] = kp * (target[i] - qpos[i]) - kd * qvel[i]
with tiny torque motors (ctrlrange +/-0.25 N*m).

External interface mirrors Go2Sim so the fly sensory adaptors work
unchanged: foot_positions and contact_forces are 4-element arrays in
[FL, FR, RL, RR] order, position is in meters, heading_angle is the yaw
of the body's forward axis (+Y for Q1 Lite, vs +X for Go2).

Usage:
    sim = Q1LiteSim('q1lite/scene.xml', timestep=0.001)
    sim.reset()
    while True:
        sim.step(joint_targets)   # 8-element array of target positions (rad)
        pos = sim.position        # world x,y,z (m)
        feet = sim.foot_positions # 4x3 world positions (m)
"""

import numpy as np
from pathlib import Path

import mujoco
from mujoco import viewer

# ============================================================================
# Constants
# ============================================================================

# 4-element foot/contact array order (matches Go2Sim / fly sensory adaptors)
FOOT_ORDER = ['FL', 'FR', 'RL', 'RR']

# Default PD gains (proven in q1lite/view_demo.py for the 0.25 N*m motors)
DEFAULT_KP = 5.0
DEFAULT_KD = 0.3


# ============================================================================
# Q1LiteSim
# ============================================================================

class Q1LiteSim:
    """
    Torque-controlled MuJoCo simulation for the Q1 Lite spider quadruped.

    Usage:
        sim = Q1LiteSim('q1lite/scene.xml')
        sim.reset()
        while True:
            sim.step(joint_targets)   # (8,) target positions in radians
            pos = sim.position        # world x,y,z (m)
            heading = sim.heading_angle  # yaw of body +Y axis (rad)
            feet = sim.foot_positions    # 4x3 world positions (m)
    """

    def __init__(self, model_path, timestep=0.001,
                 kp=None, kd=None):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

        self.kp = kp if kp is not None else DEFAULT_KP
        self.kd = kd if kd is not None else DEFAULT_KD

        # Actuator -> joint address maps (robust to any actuator wiring):
        # ctrl index <-> qpos/qvel addresses via each actuator's transmission
        nu = self.model.nu
        self._act_qposadr = np.zeros(nu, dtype=np.int64)
        self._act_dofadr = np.zeros(nu, dtype=np.int64)
        for i in range(nu):
            jid = self.model.actuator_trnid[i, 0]
            self._act_qposadr[i] = self.model.jnt_qposadr[jid]
            self._act_dofadr[i] = self.model.jnt_dofadr[jid]
        self._ctrl_limited = self.model.actuator_ctrllimited.astype(bool)
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        # Body / keyframe / foot geom lookups
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
        self._stand_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
        self._find_foot_geoms()

        # Internal state
        self._step_count = 0
        self._last_ctrl = np.zeros(nu, dtype=np.float64)

        # Visual objects (taste zones / odor sources), hidden until placed
        self._setup_visual_objects()

        # Looming ball body/joint IDs (present in q1lite/scene.xml;
        # tolerant no-op if the model has no ball)
        self._ball_body_id = -1
        self._ball_jnt_qpos_adr = -1
        self._ball_jnt_dof_adr = -1
        try:
            self._ball_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, 'looming_ball')
            ball_jnt = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, 'looming_ball_joint')
            if ball_jnt >= 0:
                self._ball_jnt_qpos_adr = int(self.model.jnt_qposadr[ball_jnt])
                self._ball_jnt_dof_adr = int(self.model.jnt_dofadr[ball_jnt])
        except Exception:
            pass  # no looming ball in scene

    def _find_foot_geoms(self):
        """Resolve the foot (box) geom inside each <leg>_upper body."""
        self._foot_geom_by_leg = {}
        for leg in ('FR', 'RR', 'FL', 'RL'):
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f'{leg}_upper')
            if bid < 0:
                raise ValueError(f"Q1LiteSim: body '{leg}_upper' not found")
            adr = self.model.body_geomadr[bid]
            num = self.model.body_geomnum[bid]
            gid = -1
            for g in range(adr, adr + num):
                # The foot is the box geom; the upper-leg collision is a
                # capsule and the visual is a mesh.
                if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    gid = g
                    break
            if gid < 0:
                raise ValueError(f"Q1LiteSim: foot geom not found for '{leg}'")
            self._foot_geom_by_leg[leg] = gid
        # geom id -> slot in FOOT_ORDER ([FL, FR, RL, RR])
        self._foot_geom_slot = {
            self._foot_geom_by_leg[leg]: i
            for i, leg in enumerate(FOOT_ORDER)}

    # ── Visual objects (taste zones / odor sources) ─────────────────────

    _TASTE_ALPHA = 0.5
    _ODOR_CORE_ALPHA = 0.7
    _ODOR_HALO_ALPHA = 0.15

    def _setup_visual_objects(self):
        """Resolve placeholder geoms/materials; hide them by default."""
        def _gid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

        def _mid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, name)

        self._taste_geom_ids = [_gid('taste_zone_0'), _gid('taste_zone_1')]
        self._odor_core_ids = [_gid('odor_core_0'), _gid('odor_core_1')]
        self._odor_halo_ids = [_gid('odor_halo_0'), _gid('odor_halo_1')]

        self._mat_taste = {'sugar': _mid('taste_sugar'),
                           'bitter': _mid('taste_bitter')}
        self._mat_odor_core = {'attractive': _mid('odor_att_core'),
                               'repulsive': _mid('odor_rep_core')}
        self._mat_odor_halo = {'attractive': _mid('odor_att_halo'),
                               'repulsive': _mid('odor_rep_halo')}

        self._hide_taste()
        self._hide_odor()

    def _mids(self, dct):
        return [m for m in dct.values() if m >= 0]

    def _hide_taste(self):
        for mid in self._mids(self._mat_taste):
            self.model.mat_rgba[mid, 3] = 0.0

    def _hide_odor(self):
        for mid in self._mids(self._mat_odor_core) + self._mids(self._mat_odor_halo):
            self.model.mat_rgba[mid, 3] = 0.0

    def place_taste_zones(self, zones):
        """Position/show taste-zone patches from TasteZone objects.

        Args:
            zones: sequence of objects with .center (x,y in mm), .radius (mm),
                   .taste ('sugar' or 'bitter').
        """
        self._hide_taste()
        for i, zone in enumerate(zones):
            if i >= len(self._taste_geom_ids):
                break
            gid = self._taste_geom_ids[i]
            if gid < 0:
                continue
            matid = self._mat_taste.get(getattr(zone, 'taste', 'sugar'), -1)
            r = float(zone.radius) / 1000.0
            self.model.geom_pos[gid] = [float(zone.center[0]) / 1000.0,
                                        float(zone.center[1]) / 1000.0, 0.02]
            self.model.geom_size[gid] = [r, 0.02, 0.0]
            if matid >= 0:
                self.model.geom_matid[gid] = matid
                self.model.mat_rgba[matid, 3] = self._TASTE_ALPHA

    def place_odor_sources(self, sources):
        """Position/show odor-source orbs + halos from OdorSource objects.

        Args:
            sources: sequence of objects with .position (x,y,z in mm),
                     .odor_type ('attractive'/'repulsive'), .spread (mm).
        """
        self._hide_odor()
        for i, src in enumerate(sources):
            if i >= len(self._odor_core_ids):
                break
            otype = getattr(src, 'odor_type', 'attractive')
            pos = src.position
            x = float(pos[0]) / 1000.0
            y = float(pos[1]) / 1000.0
            z = float(pos[2]) / 1000.0 if len(pos) > 2 else 0.3
            halo_r = max(float(src.spread) / 1000.0 * 0.5, 0.3)
            core_r = max(halo_r * 0.3, 0.15)

            cgid = self._odor_core_ids[i]
            hgid = self._odor_halo_ids[i]
            cmat = self._mat_odor_core.get(otype, -1)
            hmat = self._mat_odor_halo.get(otype, -1)
            if cgid >= 0:
                self.model.geom_pos[cgid] = [x, y, z]
                self.model.geom_size[cgid] = [core_r, 0.0, 0.0]
                if cmat >= 0:
                    self.model.geom_matid[cgid] = cmat
                    self.model.mat_rgba[cmat, 3] = self._ODOR_CORE_ALPHA
            if hgid >= 0:
                self.model.geom_pos[hgid] = [x, y, z]
                self.model.geom_size[hgid] = [halo_r, 0.0, 0.0]
                if hmat >= 0:
                    self.model.geom_matid[hgid] = hmat
                    self.model.mat_rgba[hmat, 3] = self._ODOR_HALO_ALPHA

    def set_looming_ball(self, pos, size=None):
        """Move the looming ball to a world position (no-op without ball).

        Supports both ball variants: a freejoint ball (teleported via
        qpos, velocity zeroed so gravity can't accumulate fall speed)
        and a static worldbody ball (moved via model.body_pos). Both
        call mj_forward so renders see the new position immediately.
        """
        if self._ball_jnt_qpos_adr >= 0:
            adr = self._ball_jnt_qpos_adr
            self.data.qpos[adr:adr + 3] = pos
            self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]  # identity quat
            if self._ball_jnt_dof_adr >= 0:
                self.data.qvel[self._ball_jnt_dof_adr:
                               self._ball_jnt_dof_adr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
        elif self._ball_body_id >= 0:
            self.model.body_pos[self._ball_body_id] = pos
            mujoco.mj_forward(self.model, self.data)
        if size is not None and self._ball_body_id >= 0:
            geom_id = self.model.body_geomadr[self._ball_body_id]
            if geom_id >= 0:
                self.model.geom_size[geom_id][0] = size

    def get_looming_ball_pos(self):
        """Get current looming ball position or None if not present."""
        if self._ball_body_id >= 0:
            return self.data.xpos[self._ball_body_id].copy()
        return None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self):
        """Reset simulation to the official standby stance (stand keyframe)."""
        if self._stand_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._stand_key_id)
        else:
            # Fallback: firmware standby pose (hips +/-45deg, knees 0)
            from .q1lite_adaptor import STAND_POSE
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[2] = 0.044
            self.data.qpos[7:7 + self.model.nu] = STAND_POSE
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

    # ── Simulation ────────────────────────────────────────────────────────

    def step(self, joint_targets):
        """
        Step physics by one timestep with PD torque control.

        Args:
            joint_targets: array of shape (nu,) — desired joint positions
                in radians, in actuator order (FR, RR, FL, RL x [hip, knee]).
        """
        joint_targets = np.asarray(joint_targets, dtype=np.float64).reshape(-1)
        q = self.data.qpos[self._act_qposadr]
        qd = self.data.qvel[self._act_dofadr]
        ctrl = self.kp * (joint_targets - q) - self.kd * qd
        if self._ctrl_limited.any():
            ctrl = np.where(self._ctrl_limited,
                            np.clip(ctrl, self._ctrl_lo, self._ctrl_hi), ctrl)
        self.data.ctrl[:] = ctrl
        self._last_ctrl = ctrl.copy()

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

    # ── Body State ────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        """World-frame position of base (x, y, z) in meters."""
        return self.data.qpos[0:3].copy()

    @property
    def orientation_quat(self) -> np.ndarray:
        """Body orientation quaternion (w, x, y, z)."""
        return self.data.qpos[3:7].copy()

    @property
    def forward(self) -> np.ndarray:
        """Body +Y axis (head direction) in world frame, 3-vector.

        Q1 Lite walks toward +Y (the "F" legs side), unlike Go2 whose
        forward is the body +X axis.
        """
        w, x, y, z = self.orientation_quat
        # Second column of the rotation matrix (body Y axis in world)
        fx = 2.0 * (x * y - w * z)
        fy = 1.0 - 2.0 * (x * x + z * z)
        fz = 2.0 * (y * z + w * x)
        norm = np.sqrt(fx * fx + fy * fy + fz * fz) + 1e-10
        return np.array([fx / norm, fy / norm, fz / norm])

    @property
    def heading_angle(self) -> float:
        """Yaw angle (radians) of the body +Y (forward) axis."""
        fwd = self.forward
        return float(np.arctan2(fwd[1], fwd[0]))

    @property
    def joint_positions(self) -> np.ndarray:
        """Actual joint positions in radians (actuator order)."""
        return self.data.qpos[self._act_qposadr].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        """Joint velocities in rad/s (actuator order)."""
        return self.data.qvel[self._act_dofadr].copy()

    @property
    def joint_torques(self) -> np.ndarray:
        """Last applied actuator torques (N*m, actuator order)."""
        return self._last_ctrl.copy()

    @property
    def foot_positions(self) -> np.ndarray:
        """4x3 world-frame foot positions (m), order [FL, FR, RL, RR]."""
        out = np.zeros((4, 3), dtype=np.float64)
        for i, leg in enumerate(FOOT_ORDER):
            out[i] = self.data.geom_xpos[self._foot_geom_by_leg[leg]]
        return out

    @property
    def contact_forces(self) -> np.ndarray:
        """
        4 foot contact normal-force magnitudes (N), order [FL, FR, RL, RR].

        Uses mj_contactForce on the foot box geoms. Note: Q1 Lite weighs
        ~80 g, so grounded feet carry only ~0.2 N each (vs ~40 N for Go2);
        the fly sensory adaptors scale these values appropriately.
        """
        forces = np.zeros(4, dtype=np.float64)
        cforce = np.zeros(6, dtype=np.float64)
        for ci in range(self.data.ncon):
            contact = self.data.contact[ci]
            slot = self._foot_geom_slot.get(contact.geom1)
            if slot is None:
                slot = self._foot_geom_slot.get(contact.geom2)
            if slot is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, ci, cforce)
            # Project force onto the contact normal for scalar normal force
            normal = contact.frame[0:3]
            forces[slot] += abs(np.dot(cforce[0:3], normal))
        return forces

    # ── IMU (fallbacks — q1lite.xml defines no sensors) ──────────────────

    @property
    def imu_quat(self) -> np.ndarray:
        """Body quaternion (w, x, y, z)."""
        return self.data.qpos[3:7].copy()

    @property
    def imu_gyro(self) -> np.ndarray:
        """Base angular velocity (rad/s, body-local frame for free joints)."""
        return self.data.qvel[3:6].copy()

    @property
    def imu_acc(self) -> np.ndarray:
        """Linear acceleration (placeholder — no accelerometer in model)."""
        return np.zeros(3)

    # ── Viewer ────────────────────────────────────────────────────────────

    def launch_viewer(self, title='Q1 Lite Brain-Body'):
        """Launch MuJoCo passive viewer (tracking camera on the base)."""
        self._viewer = viewer.launch_passive(
            self.model, self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        with self._viewer.lock():
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._viewer.cam.trackbodyid = self._base_body_id
            self._viewer.cam.distance = 0.5   # tiny robot: close camera
            self._viewer.cam.azimuth = -45.0
            self._viewer.cam.elevation = -25.0
        return self._viewer

    # ── Debug ─────────────────────────────────────────────────────────────

    def print_state(self):
        """Print current simulation state."""
        pos = self.position
        vel = self.data.qvel[0:3]
        print(f"  Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
        print(f"  Vel: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] m/s")
        print(f"  Heading: {np.degrees(self.heading_angle):.1f} deg")
        print(f"  Joints: {np.round(self.joint_positions, 2)}")
        print(f"  Feet z (mm): {np.round(self.foot_positions[:, 2] * 1000, 1)}")
        print(f"  Contact (N): {np.round(self.contact_forces, 2)}")
