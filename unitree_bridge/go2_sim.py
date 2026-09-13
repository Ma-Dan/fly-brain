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
                 kp=DEFAULT_KP, kd=DEFAULT_KD):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

        # PD gains
        self.kp = kp
        self.kd = kd

        # Look up body IDs for foot position queries
        self._foot_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FOOT_BODIES
        ], dtype=np.int32)

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

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

    # ── Simulation ────────────────────────────────────────────────────────

    def step(self, joint_targets):
        """
        Step physics by one timestep using PD torque control.

        Torque computation (matching DDS bridge):
            ctrl[i] = kp * (target[i] - sensordata[i])
                    - kd * sensordata[i + 12]

        Where sensordata[i] is joint position and sensordata[i+12] is velocity.

        Args:
            joint_targets: np.ndarray of shape (12,) — desired joint positions in radians.
                          Order: FR(hip,thigh,calf), FL(hip,thigh,calf),
                                 RR(hip,thigh,calf), RL(hip,thigh,calf)
        """
        # PD torque control
        cur_pos  = self.data.sensordata[0:12]
        cur_vel  = self.data.sensordata[12:24]
        self.data.ctrl[:] = (
            self.kp * (joint_targets - cur_pos) - self.kd * cur_vel
        )

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
        4 contact force magnitudes (N), estimated from foot z-height.

        Uses penetration depth as a proxy since the default Go2 scene
        has no foot touch sensors.
        """
        ground_z = 0.0
        penetration = ground_z - self._foot_positions[:, 2]
        contact = np.maximum(penetration, 0.0)
        self._contact_forces = contact * 5000.0
        return self._contact_forces.copy()

    # ── IMU ───────────────────────────────────────────────────────────────

    @property
    def imu_quat(self) -> np.ndarray:
        """IMU quaternion (w, x, y, z)."""
        return self.data.sensordata[IMU_QUAT_ADR:IMU_QUAT_ADR + 4].copy()

    @property
    def imu_gyro(self) -> np.ndarray:
        """IMU angular velocity (x, y, z) in rad/s."""
        return self.data.sensordata[IMU_GYRO_ADR:IMU_GYRO_ADR + 3].copy()

    @property
    def imu_acc(self) -> np.ndarray:
        """IMU linear acceleration (x, y, z) in m/s²."""
        return self.data.sensordata[IMU_ACC_ADR:IMU_ACC_ADR + 3].copy()

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