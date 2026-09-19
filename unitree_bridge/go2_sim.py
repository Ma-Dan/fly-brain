"""
Go2Sim: Torque-controlled MuJoCo wrapper for Unitree Go2 quadruped.

Uses PD torque control (same as unitree_sdk2 DDS bridge):
    ctrl[i] = kp * (target[i] - sensordata[i]) - kd * sensordata[i + 12]

Loads go2/scene.xml, manages physics stepping, reads sensors,
and provides a clean Python interface for brain-body integration.
"""

import numpy as np
from pathlib import Path

import mujoco
from mujoco import viewer

# Optimize CPU performance on macOS
import torch
torch.set_num_threads(min(torch.get_num_threads(), 8))

# ============================================================================
# Constants
# ============================================================================

# Joint name → actuator index mapping (from scene.xml inspection)
ACTUATOR_NAMES = [
    'FR_hip', 'FR_thigh', 'FR_calf',
    'FL_hip', 'FL_thigh', 'FL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
]

# Foot body names for position queries
FOOT_BODIES = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']

# Sensor data offsets (from sensor_adr inspection)
# Joint positions: sensordata[0:12]
# Joint velocities: sensordata[12:24]
# Joint torques: sensordata[24:36]
IMU_QUAT_ADR  = 36   # 4 floats: w,x,y,z
IMU_GYRO_ADR  = 40   # 3 floats
IMU_ACC_ADR   = 43   # 3 floats
FRAME_POS_ADR = 46   # 3 floats
FRAME_VEL_ADR = 49   # 3 floats

# Default joint angles for standing (all zeros in MJCF = default pose)
STAND_POSE = np.zeros(12, dtype=np.float64)

# Default PD gains (like Unitree DDS bridge defaults)
DEFAULT_KP = 40.0
DEFAULT_KD = 1.0


# ============================================================================
# Go2Sim
# ============================================================================

class Go2Sim:
    """
    Torque-controlled MuJoCo simulation for Unitree Go2.

    Uses PD torque control: writes computed joint torques to mj_data.ctrl,
    matching the approach used by the unitree_sdk2 DDS bridge.

    Usage:
        sim = Go2Sim(model_path='vendor/unitree_mujoco/unitree_robots/go2/scene.xml')
        sim.reset()
        while True:
            sim.step(joint_targets)   # 12-element array of target positions (rad)
            pos = sim.position        # world x,y,z (m)
            heading = sim.forward     # body X-axis in world
            feet = sim.foot_positions # 4×3 world positions (m)
            viewer.sync()
    """

    def __init__(self, model_path, timestep=0.001,
                 kp=None, kd=None):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

        # Detect actuator type: general (MJX) = built-in PD, motor (Unitree) = manual
        self._use_general_actuators = (
            self.model.nu > 0 and self.model.actuator_trntype[0] != 0)
        # Store for compatibility but not used with general actuators
        self.kp = kp if kp is not None else 50.0
        self.kd = kd if kd is not None else 1.0

        # Sensor offsets: auto-detect from naming convention
        self._detect_sensor_offsets()

        # Look up body IDs for foot position queries
        # MJX scene has no foot child bodies; use calf bodies instead
        foot_bodies_fallback = {
            'FL_foot': 'FL_calf', 'FR_foot': 'FR_calf',
            'RL_foot': 'RL_calf', 'RR_foot': 'RR_calf',
        }
        self._foot_body_ids = np.zeros(4, dtype=np.int32)
        for i, name in enumerate(FOOT_BODIES):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, foot_bodies_fallback[name])
            self._foot_body_ids[i] = bid

        # Internal state
        self._step_count = 0
        self._foot_positions = np.zeros((4, 3), dtype=np.float64)
        self._contact_forces = np.zeros(4, dtype=np.float64)

        # Visual objects (taste zones / odor sources), hidden until placed
        self._setup_visual_objects()

    def _detect_sensor_offsets(self):
        """Auto-detect sensor addresses by scanning sensor names."""
        self._imu_gyro_adr = -1
        self._imu_acc_adr = -1
        self._imu_quat_adr = -1
        self._frame_pos_adr = -1
        self._frame_vel_adr = -1

        adr_map = {
            'imu_quat': ('_imu_quat_adr', 4),
            'orientation': ('_imu_quat_adr', 4),
            'imu_gyro': ('_imu_gyro_adr', 3),
            'gyro': ('_imu_gyro_adr', 3),
            'imu_acc': ('_imu_acc_adr', 3),
            'accelerometer': ('_imu_acc_adr', 3),
            'frame_pos': ('_frame_pos_adr', 3),
            'global_position': ('_frame_pos_adr', 3),
            'frame_vel': ('_frame_vel_adr', 3),
            'global_linvel': ('_frame_vel_adr', 3),
        }
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name and name in adr_map:
                attr, dim = adr_map[name]
                setattr(self, attr, self.model.sensor_adr[i])

        # Internal state
        self._step_count = 0
        self._foot_positions = np.zeros((4, 3), dtype=np.float64)
        self._contact_forces = np.zeros(4, dtype=np.float64)

        # Looming ball body/joint IDs (look up lazily)
        self._ball_body_id = -1
        self._ball_jnt_qpos_adr = -1
        try:
            self._ball_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, 'looming_ball')
            ball_jnt = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, 'looming_ball_joint')
            if ball_jnt >= 0:
                self._ball_jnt_qpos_adr = int(self.model.jnt_qposadr[ball_jnt])
        except Exception:
            pass  # no looming ball in scene

    def set_looming_ball(self, pos, size=None):
        """Move the looming ball to a world position.
        
        Args:
            pos: (x, y, z) world position in meters
            size: optional radius override in meters
        """
        if self._ball_jnt_qpos_adr >= 0:
            adr = self._ball_jnt_qpos_adr
            self.data.qpos[adr:adr + 3] = pos
            self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]  # identity quat
        if size is not None and self._ball_body_id >= 0:
            # Update geom size
            geom_id = self.model.body_geomadr[self._ball_body_id]
            if geom_id >= 0:
                self.model.geom_size[geom_id][0] = size

    def get_looming_ball_pos(self):
        """Get current looming ball position or None if not present."""
        if self._ball_jnt_qpos_adr >= 0:
            return self.data.qpos[self._ball_jnt_qpos_adr:
                                  self._ball_jnt_qpos_adr + 3].copy()
        return None

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

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self):
        """Reset simulation to initial state (matching go2_walk.py reference)."""
        mujoco.mj_resetData(self.model, self.data)
        # Set initial stance: z=0.35, thigh=0.8, calf=-1.5, hip=0
        self.data.qpos[2] = 0.35
        self.data.qpos[7:19] = [0, 0.8, -1.5] * 4
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

    # ── Simulation ────────────────────────────────────────────────────────

    def step(self, joint_targets):
        """
        Step physics by one timestep.

        For general actuators (MJX scene): writes position targets to ctrl.
          PD servo is built-in (gainprm[0]=50, biasprm[1]=50).
        For motor actuators (standard scene): applies manual PD torque
          (kp=50, kd=1.5) matching go2_walk.py reference.

        Args:
            joint_targets: np.ndarray of shape (12,) — desired joint positions in radians.
        """
        if self._use_general_actuators:
            self.data.ctrl[:] = joint_targets
        else:
            # Manual PD torque control for motor actuators (kp=50, kd=1.5)
            kp, kd = 50.0, 1.5
            for j in range(12):
                pos = self.data.qpos[7 + j]
                vel = self.data.qvel[6 + j]
                self.data.ctrl[j] = kp * (joint_targets[j] - pos) - kd * vel

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self._update_foot_positions()

    def _update_foot_positions(self):
        """Cache world-frame positions of 4 feet from forward kinematics."""
        for i, body_id in enumerate(self._foot_body_ids):
            self._foot_positions[i] = self.data.xpos[body_id]

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
        """Body X-axis (forward direction) in world frame, 3-vector."""
        quat = self.orientation_quat
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        fx = 1 - 2 * (y * y + z * z)
        fy = 2 * (x * y + w * z)
        fz = 2 * (x * z - w * y)
        norm = np.sqrt(fx*fx + fy*fy + fz*fz) + 1e-10
        return np.array([fx/norm, fy/norm, fz/norm])

    @property
    def heading_angle(self) -> float:
        """Yaw angle (radians) derived from forward vector."""
        fwd = self.forward
        return float(np.arctan2(fwd[1], fwd[0]))

    @property
    def joint_positions(self) -> np.ndarray:
        """12 actual joint positions in radians."""
        return self.data.sensordata[0:12].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        """12 joint velocities in rad/s."""
        return self.data.sensordata[12:24].copy()

    @property
    def joint_torques(self) -> np.ndarray:
        """12 estimated joint torques (from sensor)."""
        return self.data.sensordata[24:36].copy()

    @property
    def foot_positions(self) -> np.ndarray:
        """4×3 world-frame foot positions (meters)."""
        return self._foot_positions.copy()

    @property
    def contact_forces(self) -> np.ndarray:
        """
        4 foot contact force magnitudes (N) from calf collision geoms.

        Uses mj_contactForce on FL/FR/RL/RR geoms (calf body foot
        spheres) to compute real normal forces.
        """
        forces = np.zeros(4, dtype=np.float64)
        calf_names = ['FL', 'FR', 'RL', 'RR']

        if not hasattr(self, '_calf_ids'):
            self._calf_ids = np.array([
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in calf_names], dtype=np.int32)

        cforce = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            for fi, gid in enumerate(self._calf_ids):
                if gid >= 0 and (g1 == gid or g2 == gid):
                    mujoco.mj_contactForce(self.model, self.data, i, cforce)
                    # cforce[0:3] = world-frame force on body2
                    # contact.frame[0:3] = contact normal (world frame)
                    normal = contact.frame[0:3]
                    # Project force onto contact normal for scalar normal force
                    normal_force = abs(np.dot(cforce[0:3], normal))
                    forces[fi] += normal_force
                    break
        return forces

    # ── IMU ───────────────────────────────────────────────────────────────

    @property
    def imu_quat(self) -> np.ndarray:
        """IMU quaternion (w, x, y, z)."""
        if self._imu_quat_adr >= 0:
            return self.data.sensordata[self._imu_quat_adr:self._imu_quat_adr + 4].copy()
        return self.data.qpos[3:7].copy()  # fallback to body quat

    @property
    def imu_gyro(self) -> np.ndarray:
        """IMU angular velocity (x, y, z) in rad/s."""
        if self._imu_gyro_adr >= 0:
            return self.data.sensordata[self._imu_gyro_adr:self._imu_gyro_adr + 3].copy()
        return self.data.qvel[3:6].copy()

    @property
    def imu_acc(self) -> np.ndarray:
        """IMU linear acceleration (x, y, z) in m/s²."""
        if self._imu_acc_adr >= 0:
            return self.data.sensordata[self._imu_acc_adr:self._imu_acc_adr + 3].copy()
        return np.zeros(3)

    # ── Viewer ────────────────────────────────────────────────────────────

    def launch_viewer(self, title='Go2 Brain-Body'):
        """Launch MuJoCo passive viewer."""
        self._viewer = viewer.launch_passive(
            self.model, self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        with self._viewer.lock():
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._viewer.cam.trackbodyid = 1  # base_link
            self._viewer.cam.distance = 2.5
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
        print(f"  Heading: {np.degrees(self.heading_angle):.1f}°")
        print(f"  Joints: {np.round(self.joint_positions, 2)}")
        print(f"  Feet z: {np.round(self._foot_positions[:, 2], 3)}")
        print(f"  Contact: {np.round(self.contact_forces, 1)}")