"""
Go2Adaptor: Converts DN firing rates to Go2 joint position targets.

Uses sinusoidal CPG (go2_walk.py reference) with manual PD torque control.
Standard Menagerie scene: motor actuators, friction=0.8, damping=2.0.
"""

import numpy as np
import math

ACTUATOR_PER_LEG = 3

# Standing pose (matching go2_walk.py: thigh=0.8, calf=-1.5, hip=0)
#
# Thigh sign convention (verified against go2.xml forward kinematics):
#   POSITIVE thigh = leg swings BACKWARD (caudal), NEGATIVE = forward.
# At thigh=0.8 the foot sits ~0.01 m behind its hip for all four legs, so the
# standing pose is symmetric; the RL forward-drift compensation lives in
# QuadCPG.step (see RL_FORWARD_DRIFT_BIAS) rather than as a magic stand offset.
STAND_HIP = 0.0
STAND_THIGH = 0.8
STAND_KNEE = -1.5
STAND_POSE = np.array([
    STAND_HIP, STAND_THIGH, STAND_KNEE,      # FL
    STAND_HIP, STAND_THIGH, STAND_KNEE,      # FR
    STAND_HIP, STAND_THIGH, STAND_KNEE,      # RL
    STAND_HIP, STAND_THIGH, STAND_KNEE,      # RR
], dtype=np.float64)

# RL thigh oscillation-mean bias (rad, POSITIVE = swing backward) that holds
# the rear-left foot under its hip and cancels the residual trot yaw. Set to 0
# to disable.
RL_FORWARD_DRIFT_BIAS = 0.4


class QuadCPG:
    """
    Sinusoidal trot CPG matching go2_walk.py reference.

    Thigh: sin oscillation (push/pull)
    Calf: cos oscillation (foot lift, π/2 phase-shifted from thigh)
    6 Hz base frequency, [0, π, π, 0] phase offsets for diagonal trot.
    """

    def __init__(self, dt=0.001, freq=8.0, thigh_amp=0.7, calf_amp=0.35):
        self.dt = dt
        self.freq = freq
        self.thigh_amp = thigh_amp
        self.calf_amp = calf_amp
        self.stand_offsets = STAND_POSE.copy()
        self.step_count = 0
        # Trot phase offsets: FR=0, FL=π, RR=π, RL=0 (diagonal pairs sync)
        self.phases = np.array([0.0, math.pi, math.pi, 0.0])

    def step(self, forward_drive, turn_drive):
        forward_drive = np.clip(forward_drive, 0.0, 1.0)
        turn_drive = np.clip(turn_drive, -1.0, 1.0)

        if forward_drive < 0.01:
            return self.stand_offsets.copy()

        # Drive-modulated amplitude: nonlinear for better mid-range response
        amp = min(forward_drive * 1.3, 1.0)  # drive=0.5 → amp=0.65
        thigh_a = self.thigh_amp * amp
        calf_a = self.calf_amp * amp
        f = self.freq * (0.3 + 0.7 * forward_drive)

        t = self.step_count * self.dt * f * 2 * math.pi

        targets = self.stand_offsets.copy()
        # FL(0-2), FR(3-5), RL(6-8), RR(9-11)
        for leg in range(4):
            base = leg * ACTUATOR_PER_LEG
            phase = self.phases[leg]
            side_scale = (1.0 + turn_drive) if leg in [1, 2] else (1.0 - turn_drive)
            ta = thigh_a * np.clip(side_scale, 0.3, 1.7)

            # Oscillate around per-leg stand position (supports custom RL offset)
            hip0  = self.stand_offsets[base + 0]
            thigh0 = self.stand_offsets[base + 1]
            calf0  = self.stand_offsets[base + 2]

            # RL forward-drift compensation.
            #
            # During the trot the RL (rear-left) foot creeps forward relative
            # to the body, which tips the rear-left support and induces a yaw
            # during nominally-straight walking. Because POSITIVE thigh swings
            # the leg BACKWARD, the compensation must be a POSITIVE bias on the
            # RL thigh so its oscillation mean shifts backward and holds the
            # foot under the hip. (A negative bias pushes RL *forward* and makes
            # the drift/turn worse — this was previously inverted.)
            bias = RL_FORWARD_DRIFT_BIAS if leg == 2 else 0.0  # leg 2 = RL

            targets[base + 0] = hip0
            targets[base + 1] = thigh0 + ta * (math.sin(t + phase) + bias)
            targets[base + 2] = calf0 + calf_a * math.cos(t + phase)

        self.step_count += 1
        return targets

    def reset(self):
        self.step_count = 0


class Go2Adaptor:
    """Brain → Go2 joint targets."""

    def __init__(self, decoder, dt=0.01):
        from brain_body_bridge import BrainBodyBridge
        self.bridge = BrainBodyBridge(decoder)
        self.cpg = QuadCPG(dt=0.001)
        self.dt = dt
        self.drive = np.array([0.0, 0.0])
        self.mode = 'walking'
        self.drive_gain = 8.0   # brain ~0.12 → CPG drive ~1.0 for full speed
        self.turn_gain = 2.0
        self._escape_cooldown = 0.0
        self._escape_min_gap = 2.0

    def compute_drive(self, dt=None):
        if dt is None: dt = self.dt
        self._escape_cooldown = max(0.0, self._escape_cooldown - dt)
        self.drive = self.bridge.compute_drive(dt=dt)
        mode = self.bridge.mode
        if mode == 'escape' and self._escape_cooldown > 0: mode = 'walking'
        elif mode == 'escape': self._escape_cooldown = self._escape_min_gap
        self.mode = mode

        # bridge.compute_drive() returns a DIFFERENTIAL drive [left, right]
        # (left ≈ right ≈ forward when walking straight). Derive forward from
        # their mean and turn from their left/right asymmetry. Reading drive[0]
        # as forward and drive[1] as turn injects a spurious constant turn
        # (~ right_drive * gain) during straight walking.
        left, right = self.drive[0], self.drive[1]
        if mode == 'escape':
            fwd = np.clip((abs(left) + abs(right)) * 0.5 * self.drive_gain, 0.0, 1.0)
            turn = np.clip((left - right) * 2.0 * self.turn_gain, -1.0, 1.0)
            # A looming threat straight ahead has no L/R cue (turn ≈ 0), but
            # running straight forward would run *into* the approaching ball.
            # Dodge by turning hard (veering off) instead of charging at it.
            if abs(turn) < 0.2:
                turn = 1.0
                fwd = min(fwd, 0.5)
        elif mode in ('grooming', 'feeding'):
            fwd, turn = 0.0, 0.0
        else:
            fwd = np.clip((left + right) * 0.5 * self.drive_gain, 0.0, 1.0)
            turn = np.clip((left - right) * self.turn_gain, -1.0, 1.0)
        return fwd, turn

    def compute_action(self, dt=None):
        return self.cpg.step(*self.compute_drive(dt))

    def reset(self):
        self.cpg.reset()
        self.drive[:] = 0.0
        self.mode = 'walking'

    def get_status_str(self):
        d = self.drive
        return f"[{self.mode:>8s}] drive=[{d[0]:.2f}, {d[1]:.2f}] phases={np.round(self.cpg.phases, 2)}"