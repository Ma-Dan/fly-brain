"""
Q1LiteAdaptor: Converts DN firing rates to Q1 Lite joint position targets.

Q1 Lite (Jason Workshop) is an 8-DOF spider quadruped:
    hip  = YAW about Z (inboard servo)  — swings the foot fore/aft
    knee = LIFT about Y (corner servo)  — raises/lowers the foot

Gait: the spider trot proven in q1lite/view_demo.py:
    q_hip  = HIP0 + SIDE * A_HIP * sin(2*pi*f*t + phase)
    q_knee = A_KNEE * (0.5 - 0.5*cos(2*pi*f*t + phase))
Left legs (FL/RL) have their frames rotated 180 deg about Z (local +X =
world -X), so their hip oscillation is sign-mirrored (SIDE) versus the
right legs (FR/RR). Positive knee lifts the foot; the foot is down at
phase 0 (start of stance push) and highest at phase pi (mid-swing).

Brain mapping (same contract as the Go2 bridge):
    forward_drive (0..1)  -> trot amplitude + frequency
    turn_drive    (-1..1) -> differential side amplitude + uniform hip yaw
                             bias. turn > 0 = veer right (CW), turn < 0 =
                             veer left (CCW).

Turn mechanics: the hips are yaw joints, so a *uniform* hip bias leans the
left legs forward and the right legs backward at the same time (positive
hip = forward for right legs, backward for left legs). Left legs leaning
forward push longer during stance -> body yaws CW (veer right). The
differential amplitude (left strides longer, right shorter) reinforces the
same direction.

Actuator order (q1lite.xml):
    FR_hip FR_knee  RR_hip RR_knee  FL_hip FL_knee  RL_hip RL_knee
"""

import numpy as np
import math

ACTUATOR_PER_LEG = 2

# q1lite.xml actuator order
LEGS = ['FR', 'RR', 'FL', 'RL']

# Standby stance (official firmware pose, model-local radians):
#   FR +45deg, RR -45deg, FL -45deg, RL +45deg, knees 0 (feet down)
HIP0 = {'FR': +0.785, 'RR': -0.785, 'FL': -0.785, 'RL': +0.785}

# Trot: diagonal pairs FR+RL / RR+FL move together
PHASE = {'FR': 0.0, 'RR': math.pi, 'FL': math.pi, 'RL': 0.0}

# Left-right mirror: left legs oscillate with opposite sign (+sin vs -sin)
SIDE = {'FR': -1.0, 'RR': -1.0, 'FL': +1.0, 'RL': +1.0}

RIGHT_LEGS = ('FR', 'RR')
LEFT_LEGS = ('FL', 'RL')

# Hip joint range is +/-1.7 rad; keep margin so oscillation + turn bias
# never ride the joint limit.
HIP_LIMIT = 1.6

STAND_POSE = np.array([
    HIP0['FR'], 0.0,
    HIP0['RR'], 0.0,
    HIP0['FL'], 0.0,
    HIP0['RL'], 0.0,
], dtype=np.float64)


class QuadCPG:
    """
    Spider trot CPG matching q1lite/view_demo.py.

    Hip:  yaw oscillation around the standby angle (fore/aft stride),
          sign-mirrored on the left legs via SIDE.
    Knee: lift oscillation, 0 at stance start (foot planted), max at
          mid-swing. Phase-locked to the hip (cos vs sin).
    1.3 Hz base frequency, diagonal-pair phases [0, pi, pi, 0].
    """

    def __init__(self, dt=0.001, freq=1.3, hip_amp=0.4, knee_amp=0.6,
                 turn_bias_gain=0.25, turn_amp_gain=0.5):
        """
        Args:
            dt: physics timestep (s). CPG is stepped once per physics step.
            freq: base trot frequency (Hz) at full drive (view_demo value).
            hip_amp: hip yaw oscillation amplitude (rad) at full drive.
            knee_amp: knee lift oscillation amplitude (rad) at full drive.
            turn_bias_gain: uniform hip yaw bias (rad) at full turn drive.
            turn_amp_gain: differential stride scaling at full turn drive.
        """
        self.dt = dt
        self.freq = freq
        self.hip_amp = hip_amp
        self.knee_amp = knee_amp
        self.turn_bias_gain = turn_bias_gain
        self.turn_amp_gain = turn_amp_gain
        self.stand_offsets = STAND_POSE.copy()
        self.step_count = 0
        self._phase_t = 0.0  # phase accumulator (rad)
        self.phases = np.array([PHASE[leg] for leg in LEGS])
        self.sides = np.array([SIDE[leg] for leg in LEGS])

    def step(self, forward_drive, turn_drive):
        """
        Compute joint position targets for one physics step.

        Args:
            forward_drive: 0..1 forward locomotion drive.
            turn_drive: -1..1; >0 = veer right (CW), <0 = veer left (CCW).

        Returns:
            np.ndarray of shape (8,) — desired joint positions in radians,
            in q1lite.xml actuator order (FR, RR, FL, RL × [hip, knee]).
        """
        forward_drive = np.clip(forward_drive, 0.0, 1.0)
        turn_drive = np.clip(turn_drive, -1.0, 1.0)

        if forward_drive < 0.01:
            return self.stand_offsets.copy()

        # Drive-modulated amplitude (same shape as the Go2 CPG):
        # drive=0.5 -> amp=0.65, drive>=0.77 -> full stride
        amp = min(forward_drive * 1.3, 1.0)
        f = self.freq * (0.35 + 0.65 * forward_drive)

        # Phase accumulator: the drive (and thus frequency) changes every
        # brain bundle; accumulating phase avoids the discontinuities a
        # step_count-based clock would cause when f changes mid-stride.
        self._phase_t += self.dt * f * 2.0 * math.pi
        t = self._phase_t

        targets = self.stand_offsets.copy()
        for i, leg in enumerate(LEGS):
            base = i * ACTUATOR_PER_LEG
            th = t + self.phases[i]

            # Differential stride: turn>0 (veer right) -> left legs step
            # longer, right legs shorter.
            if leg in RIGHT_LEGS:
                side_amp = amp * np.clip(
                    1.0 - self.turn_amp_gain * turn_drive, 0.4, 1.6)
            else:
                side_amp = amp * np.clip(
                    1.0 + self.turn_amp_gain * turn_drive, 0.4, 1.6)

            # Uniform yaw bias: -bias leans right legs backward and left
            # legs forward (positive hip = forward for right legs,
            # backward for left legs), yawing the body CW for turn > 0.
            hip = (HIP0[leg]
                   + self.sides[i] * self.hip_amp * side_amp * math.sin(th)
                   - self.turn_bias_gain * turn_drive)

            # Knee lift follows global amplitude only (uniform foot
            # clearance regardless of turning).
            knee = self.knee_amp * amp * (0.5 - 0.5 * math.cos(th))

            targets[base + 0] = np.clip(hip, -HIP_LIMIT, HIP_LIMIT)
            targets[base + 1] = knee

        self.step_count += 1
        return targets

    def reset(self):
        self.step_count = 0
        self._phase_t = 0.0


class Q1LiteAdaptor:
    """Brain → Q1 Lite joint targets."""

    def __init__(self, decoder, dt=0.01, bridge_kwargs=None):
        """
        Args:
            decoder: DNRateDecoder instance.
            dt: brain-bundle interval (s).
            bridge_kwargs: extra kwargs for BrainBodyBridge. Q1 Lite is
                ~190x lighter than Go2, so tactile_escape_force must be
                passed in *scaled* newtons (see fly_embodied_q1lite.py).
        """
        from brain_body_bridge import BrainBodyBridge
        self.bridge = BrainBodyBridge(decoder, **(bridge_kwargs or {}))
        self.cpg = QuadCPG(dt=0.001)
        self.dt = dt
        self.drive = np.array([0.0, 0.0])
        self.mode = 'walking'
        self.drive_gain = 8.0   # brain ~0.12 -> CPG drive ~1.0 for full speed
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
        # (left ~ right ~ forward when walking straight). Derive forward
        # from their mean and turn from their left/right asymmetry.
        left, right = self.drive[0], self.drive[1]
        if mode == 'escape':
            fwd = np.clip((abs(left) + abs(right)) * 0.5 * self.drive_gain, 0.0, 1.0)
            turn = np.clip((left - right) * 2.0 * self.turn_gain, -1.0, 1.0)
            # A threat straight ahead has no L/R cue (turn ~ 0), but running
            # straight forward would run *into* it. Dodge by veering hard
            # instead of charging ahead.
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
