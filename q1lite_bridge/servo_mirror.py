"""
ServoMirror: streams ONE CPG joint command to a real PWM servo (MG90D)
on a Raspberry Pi while the brain-body simulation runs.

Used by fly_embodied_q1lite.py --servo. Reuses /home/ubuntu/servo_stream.py
on the Pi (stdin angle stream -> pigpio, GPIO 18). send() is internally
rate-limited to ~50Hz so callers can call it every physics step.

Angle mapping: servo_deg = center + scale * (cmd_rad - stand_rad) —
the joint's standby pose maps to `center` (default 90 deg, mechanical
mid-range); swings map 1:1 in degrees. Negative `scale` for reversed
servo mounting.
"""

import subprocess
import time

from .q1lite_adaptor import HIP0, LEGS

JOINTS = [f'{leg}_{part}' for leg in LEGS for part in ('hip', 'knee')]


def _joint_index(name):
    leg, part = name.split('_')
    return LEGS.index(leg) * 2 + (0 if part == 'hip' else 1)


def _stand_rad(name):
    leg, part = name.split('_')
    return HIP0[leg] if part == 'hip' else 0.0


class ServoMirror:
    """Mirror one simulated joint onto a real servo over SSH."""

    def __init__(self, joint='FR_hip', host='ubuntu@192.168.1.29',
                 remote_python='/home/ubuntu/miniconda3/envs/lerobot/bin/python',
                 remote_script='/home/ubuntu/servo_stream.py',
                 center=90.0, scale=57.29578, rate_hz=50.0):
        """
        Args:
            joint: joint name, e.g. 'FR_hip' or 'RR_knee'.
            host: SSH target running servo_stream.py.
            remote_python: absolute python path on the Pi.
            remote_script: servo_stream.py path on the Pi.
            center: servo angle (deg) at the joint's standby pose.
            scale: servo deg per rad of joint motion.
            rate_hz: max command rate (analog PWM servos: 50 Hz).
        """
        self.joint = joint
        self.j = _joint_index(joint)
        self.stand = _stand_rad(joint)
        self.center = float(center)
        self.scale = float(scale)
        self._interval = 1.0 / float(rate_hz)
        self._next_t = time.time()

        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
               host, f'{remote_python} {remote_script}']
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        ready = self.proc.stdout.readline().strip()
        if ready != 'ready':
            self.proc.kill()
            raise RuntimeError(f"servo stream failed on {host}: {ready!r}")

        # Park at center and let the servo settle before the loop starts
        self._send_deg(self.center)
        time.sleep(1.0)

    def send(self, targets):
        """Send a CPG target vector (nu,); internally rate-limited.

        Args:
            targets: full joint-command array (actuator order); the
                mirrored joint's command is extracted and mapped to deg.
        """
        now = time.time()
        if now < self._next_t:
            return
        self._next_t = now + self._interval
        deg = self.center + self.scale * (targets[self.j] - self.stand)
        self._send_deg(deg)

    def _send_deg(self, deg):
        deg = max(0.0, min(180.0, float(deg)))
        self.proc.stdin.write(f"{deg:.1f}\n")
        self.proc.stdin.flush()

    def close(self):
        """Park at center, stop pulsing, close the SSH stream."""
        try:
            self._send_deg(self.center)
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
