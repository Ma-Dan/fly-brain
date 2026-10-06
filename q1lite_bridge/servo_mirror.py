"""
ServoMirror: streams CPG joint commands to real PWM servos (MG90D) on a
Raspberry Pi while the brain-body simulation runs.

Used by fly_embodied_q1lite.py --servo and q1lite_servo_bridge.py.
Reuses /home/ubuntu/servo_stream.py on the Pi (stdin angle stream ->
PCA9685 over I2C). Channel i = actuator i (q1lite.xml order), so all 8
joints can be mirrored at once; each mirrored joint maps its command to
servo degrees via a per-joint center/scale:

    servo_deg = centers[joint] + scales[joint] * (cmd_rad - stand_rad)

Unmirrored channels are parked at their center (DEFAULT_CENTER).
send() is internally rate-limited to ~50Hz so callers can call it
every physics step. Negative `scale` for reversed servo mounting.
"""

import subprocess
import time

import numpy as np

from .q1lite_adaptor import HIP0, LEGS

JOINTS = [f'{leg}_{part}' for leg in LEGS for part in ('hip', 'knee')]

DEFAULT_CENTER = 90.0
DEFAULT_SCALE = 57.29578  # servo deg per rad of joint motion (1:1)


def _joint_index(name):
    leg, part = name.split('_')
    return LEGS.index(leg) * 2 + (0 if part == 'hip' else 1)


def _stand_rad(name):
    leg, part = name.split('_')
    return HIP0[leg] if part == 'hip' else 0.0


def _per_joint(value, joints, default):
    """Normalize a scalar/dict/list per-joint spec to a dict over joints.

    dict: missing keys fall back to `default`; list/tuple: aligned with
    `joints`; scalar: applies to every mirrored joint.
    """
    if isinstance(value, dict):
        return {j: float(value.get(j, default)) for j in joints}
    if isinstance(value, (list, tuple)):
        return {j: float(v) for j, v in zip(joints, value)}
    return {j: float(value) for j in joints}


def map_targets(targets, joints, centers, scales):
    """Map full 8-vector rad targets to 8 servo angles (deg, channels 0-7).

    Mirrored joint j (channel JOINTS.index(j)): centers[j] + scales[j] *
    (targets[j] - stand_rad(j)). Unmirrored channels: DEFAULT_CENTER
    (parked). Result clamped to [0, 180]. centers/scales: scalar (applies
    to all mirrored joints), dict {joint: value} (missing keys fall back
    to DEFAULT_*), or list aligned with joints.
    Returns np.ndarray shape (8,).
    """
    degs = np.full(len(JOINTS), DEFAULT_CENTER)
    cs = _per_joint(centers, joints, DEFAULT_CENTER)
    ss = _per_joint(scales, joints, DEFAULT_SCALE)
    for j in joints:
        ch = _joint_index(j)
        deg = cs[j] + ss[j] * (targets[ch] - _stand_rad(j))
        degs[ch] = max(0.0, min(180.0, deg))
    return degs


class ServoMirror:
    """Mirror simulated joints onto real servos over SSH (PCA9685, 8ch)."""

    def __init__(self, joints=None, host='ubuntu@192.168.1.141',
                 remote_python='/home/ubuntu/miniconda3/envs/lerobot/bin/python',
                 remote_script='/home/ubuntu/servo_stream.py',
                 centers=DEFAULT_CENTER, scales=DEFAULT_SCALE,
                 rate_hz=50.0, joint=None):
        """
        Args:
            joints: joint names to mirror (default: all 8, JOINTS).
            host: SSH target running servo_stream.py.
            remote_python: absolute python path on the Pi.
            remote_script: servo_stream.py path on the Pi.
            centers: servo angle (deg) at each joint's standby pose —
                scalar, {joint: deg} dict, or list aligned with joints.
            scales: servo deg per rad of joint motion — same forms;
                negative for reversed servo mounting.
            rate_hz: max command rate (analog PWM servos: 50 Hz).
            joint: legacy single-joint shorthand for joints=[joint].
        """
        if joints is None and joint is not None:
            joints = [joint]
        if joints is None:
            joints = list(JOINTS)
        joints = list(joints)
        for j in joints:
            if j not in JOINTS:
                raise ValueError(f"unknown joint {j!r} (expected one of {JOINTS})")
        self.joints = joints
        self.channels = {j: _joint_index(j) for j in joints}
        self.centers = _per_joint(centers, joints, DEFAULT_CENTER)
        self.scales = _per_joint(scales, joints, DEFAULT_SCALE)
        self._interval = 1.0 / float(rate_hz)
        self._next_t = time.time()

        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
               # keepalives: detect a dead Pi link in ~15s instead of
               # hanging on a silently-dropped WiFi connection
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3',
               host, f'{remote_python} {remote_script}']
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        ready = self.proc.stdout.readline().strip()
        if ready != f'ready {len(JOINTS)}':
            self.proc.kill()
            raise RuntimeError(f"servo stream failed on {host}: {ready!r}")

        # Park every channel at its center and let the servos settle
        # before the loop starts
        self._send_deg(self._park_degs())
        time.sleep(1.0)

    def _park_degs(self):
        """8-vector of center angles: mirrored joints at their centers,
        unmirrored channels at DEFAULT_CENTER."""
        degs = np.full(len(JOINTS), DEFAULT_CENTER)
        for j in self.joints:
            degs[self.channels[j]] = self.centers[j]
        return degs

    def send(self, targets):
        """Send a CPG target vector (nu,); internally rate-limited.

        Args:
            targets: full joint-command array (actuator order); each
                mirrored joint's command is mapped to servo degrees.
        """
        now = time.time()
        if now < self._next_t:
            return
        self._next_t = now + self._interval
        self._send_deg(map_targets(targets, self.joints,
                                   self.centers, self.scales))

    def _send_deg(self, degs):
        line = " ".join(f"{max(0.0, min(180.0, float(d))):.1f}" for d in degs)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def close(self):
        """Park at centers, stop pulsing, close the SSH stream."""
        try:
            self._send_deg(self._park_degs())
            time.sleep(0.8)
            self.proc.stdin.write("q\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
