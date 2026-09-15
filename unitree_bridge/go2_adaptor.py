"""
Go2Adaptor: Converts DN firing rates to Go2 joint position targets.

Pipeline:
  DNRateDecoder.rates → BrainBodyBridge.compute_drive() → [L, R] drive
  → QuadCPG → 12 joint targets (PD position control)

The QuadCPG implements a coordinated trot gait:
  - Diagonal pairs (FR+RL, FL+RR) alternate stance push / swing recovery
  - Stance (push): thigh goes NEGATIVE to propel body FORWARD
  - Swing (recovery): thigh holds or returns to stand, knee lifts foot
  - Drive modulates push amplitude and frequency
"""

import numpy as np

# ============================================================================
# Actuator index mapping (matches Go2 scene.xml actuator order)
# ============================================================================

ACTUATOR_PER_LEG = 3  # hip, thigh, calf

# Trot gait: diagonal pairs in phase
# FR + RL = group A, FL + RR = group B, 180° apart
TROT_PHASES = np.array([0.0, np.pi, np.pi, 0.0])  # FR, FL, RR, RL

# Standing pose (from PD-torque equilibrium at kp=40, kd=1)
STAND_THIGH = 0.3
STAND_KNEE = -1.0
# MJX actuator order: FL→FR→RL→RR
STAND_POSE = np.array(
    [0.0, STAND_THIGH, STAND_KNEE,  # FL
     0.0, STAND_THIGH, STAND_KNEE,  # FR
     0.0, STAND_THIGH, STAND_KNEE,  # RL
     0.0, STAND_THIGH, STAND_KNEE], # RR
    dtype=np.float64)


# ============================================================================
# QuadCPG — 4-leg trot oscillator (empirically tuned for Go2)
# ============================================================================

class QuadCPG:
    """
    Coordinated 4-leg trot CPG for Unitree Go2.

    Uses empirically-verified absolute push pattern:
      - Diagonal A (FR+RL) pushes thigh -0.5 rad during half-cycle
      - Diagonal B (FL+RR) pushes during the other half
      - Opposite pair lifts feet during push
      - Drive modulates amplitude linearly

    Pattern verified: ~0.15 m/s forward at drive=1.0, kp=40, kd=1.
    """

    def __init__(
        self,
        dt=0.01,                # Control timestep
        base_freq=2.0,          # Hz
        push_amp=0.5,           # rad — absolute thigh push-back
        lift_amp=0.35,          # rad — knee bend for foot lift
    ):
        self.dt = dt
        self.base_freq = base_freq
        self.push_amp = push_amp
        self.lift_amp = lift_amp

        self.stand_offsets = STAND_POSE.copy()
        self.phases = np.zeros(4, dtype=np.float64)
        self.step_count = 0

    def step(self, forward_drive, turn_drive):
        forward_drive = np.clip(forward_drive, 0.0, 1.0)
        turn_drive = np.clip(turn_drive, -1.0, 1.0)

        if forward_drive < 0.01:
            return self.stand_offsets.copy()

        amp = forward_drive

        # Turning: reduce push on inside-turn side
        left_amp = amp * (1.0 - turn_drive * 0.3)
        right_amp = amp * (1.0 + turn_drive * 0.3)

        # Frequency scales with drive
        freq = self.base_freq * (0.3 + 0.7 * forward_drive)
        self.phases += freq * 2 * np.pi * self.dt
        self.phases %= 2 * np.pi

        phase = self.phases[0]

        # Diagonal A (FL + RR): push when sine non-negative
        push_FL = -self.push_amp * left_amp if np.sin(phase) >= 0 else 0.0
        push_RR = -self.push_amp * right_amp if np.sin(phase + np.pi) < 0 else 0.0
        # Diagonal B (FR + RL): push when sine negative (opposite half-cycle)
        push_FR = -self.push_amp * right_amp if np.sin(phase + np.pi) >= 0 else 0.0
        push_RL = -self.push_amp * left_amp if np.sin(phase) < 0 else 0.0

        targets = self.stand_offsets.copy()
        # MJX actuator order: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
        targets[1] += push_FL; targets[10] += push_RR
        targets[4] += push_FR; targets[7] += push_RL

        # Foot lift on opposite pair
        if push_FL < 0 and push_RR < 0:
            targets[5] -= self.lift_amp * amp; targets[8] -= self.lift_amp * amp  # FR,RL
        if push_FR < 0 and push_RL < 0:
            targets[2] -= self.lift_amp * amp; targets[11] -= self.lift_amp * amp  # FL,RR

        self.step_count += 1
        return targets

    def reset(self):
        """Reset CPG phases."""
        self.phases = np.zeros(4, dtype=np.float64)
        self.step_count = 0


# ============================================================================
# Go2Adaptor — Brain → Body bridge
# ============================================================================

class Go2Adaptor:
    """
    Top-level adaptor: brain activity → Go2 joint targets.

    Wraps BrainBodyBridge (DN rate → drive) and QuadCPG (drive → joints).
    """

    def __init__(self, decoder, dt=0.01):
        from brain_body_bridge import BrainBodyBridge
        self.bridge = BrainBodyBridge(decoder)
        self.cpg = QuadCPG(dt=0.001)  # run CPG at physics rate (1kHz)
        self.dt = dt

        self.drive = np.array([0.0, 0.0])
        self.mode = 'walking'

        # Drive amplification: brain signals are weak (~0.1-0.3)
        # but CPG needs 0.3-0.5 for visible walking.
        self.drive_gain = 3.0

        # Escape hysteresis: prevent rapid escape/walk oscillations
        self._escape_cooldown = 0.0   # seconds remaining before re-entering escape
        self._escape_min_gap = 2.0    # minimum seconds between escape episodes

    def compute_drive(self, dt=None):
        """Compute [forward, turn] drive from DN rates (no CPG step)."""
        if dt is None:
            dt = self.dt

        self._escape_cooldown = max(0.0, self._escape_cooldown - dt)
        self.drive = self.bridge.compute_drive(dt=dt)
        mode = self.bridge.mode

        if mode == 'escape' and self._escape_cooldown > 0:
            mode = 'walking'
        elif mode == 'escape' and self._escape_cooldown <= 0:
            self._escape_cooldown = self._escape_min_gap

        self.mode = mode

        if mode == 'escape':
            fwd = min(abs(self.drive[0]) + abs(self.drive[1]), 1.0) * self.drive_gain
            fwd = min(fwd, 1.0)
            turn = np.clip(self.drive[1] * 2.0 * self.drive_gain, -1.0, 1.0)
        elif mode in ('grooming', 'feeding'):
            fwd, turn = 0.0, 0.0
        else:
            fwd = min(abs(self.drive[0]) * self.drive_gain, 1.0)
            turn = np.clip(self.drive[1] * self.drive_gain, -1.0, 1.0)

        return fwd, turn

    def compute_action(self, dt=None):
        """Deprecated: kept for compatibility. Use compute_drive() + cpg.step()."""
        fwd, turn = self.compute_drive(dt)
        return self.cpg.step(fwd, turn)

    def reset(self):
        self.cpg.reset()
        self.drive = np.array([0.0, 0.0])
        self.mode = 'walking'

    def get_status_str(self):
        d = self.drive
        return (f"[{self.mode:>8s}] drive=[{d[0]:.2f}, {d[1]:.2f}] "
                f"phases={np.round(self.cpg.phases, 2)}")