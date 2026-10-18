"""
ServoMirror: streams CPG joint commands to real PWM servos (MG90D) while
the brain-body simulation runs.

Used by fly_embodied_q1lite.py --servo and q1lite_servo_bridge.py.
Two transports share one line protocol (stdin/stdout over SSH to the Pi
running /home/ubuntu/servo_stream.py -> PCA9685 over I2C, or a raw TCP
socket to an ESP32): connect, read "ready 8", write lines of 8
space-separated servo angles (deg, channels 0-7), write "q" to stop.
Channel i = actuator i (q1lite.xml order), so all 8 joints can be
mirrored at once; each mirrored joint maps its command to servo degrees
via a per-joint center/scale:

    servo_deg = centers[joint] + scales[joint] * (cmd_rad - stand_rad)

Unmirrored channels are parked at their center (DEFAULT_CENTER).
send() is internally rate-limited to ~50Hz so callers can call it
every physics step. Negative `scale` for reversed servo mounting.
"""

import socket
import subprocess
import time

import numpy as np

from .q1lite_adaptor import HIP0, LEGS

JOINTS = [f'{leg}_{part}' for leg in LEGS for part in ('hip', 'knee')]

DEFAULT_CENTER = 90.0
DEFAULT_SCALE = 57.29578  # servo deg per rad of joint motion (1:1)

# Joints whose servo is mounted reversed on this robot — the default
# scale is negated so positive joint motion drives the servo the right
# way.  Per-joint --servo-map / --map overrides still win.
# All 4 hips: the physical hip servos sweep opposite to the sim's
# positive-hip direction (forward stride), so without the flip MuJoCo
# walking forward drives the real robot backward.
# RR_knee / FL_knee: individually reversed mounting.
MOUNT_FLIP = {'FR_hip', 'RR_hip', 'FL_hip', 'RL_hip',
              'RR_knee', 'FL_knee'}

# Knee splay at standby: the knee center is offset from vertical (90°)
# so the legs angle outward — wider stance, lower body.  The offset
# direction in servo space follows MOUNT_FLIP so all four knees splay
# the same physical direction (outward).
KNEE_SPLAY_DEG = 30.0  # >0 = outward splay; negate if legs tuck inward


def default_center(joint):
    """Default servo center (deg) at the joint's standby pose.

    Hips: DEFAULT_CENTER (90°).  Knees: offset by KNEE_SPLAY_DEG so
    the legs splay outward — the servo-space offset follows MOUNT_FLIP
    so all four knees bend the same physical way.
    """
    if joint.endswith('_knee'):
        if joint in MOUNT_FLIP:
            return DEFAULT_CENTER - KNEE_SPLAY_DEG
        return DEFAULT_CENTER + KNEE_SPLAY_DEG
    return DEFAULT_CENTER


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
    """Mirror simulated joints onto real servos (PCA9685, 8ch).

    Two transports, identical wire protocol:
      * ``transport='ssh'`` (default): SSH to a Raspberry Pi running
        ``servo_stream.py`` over stdin/stdout (unchanged legacy path).
      * ``transport='tcp'``: raw TCP socket to an ESP32 speaking the
        same line protocol — connect, read "ready 8", write 8-float angle
        lines, "q" to stop.
    """

    def __init__(self, joints=None, host='ubuntu@192.168.1.141',
                 remote_python='/home/ubuntu/miniconda3/envs/lerobot/bin/python',
                 remote_script='/home/ubuntu/servo_stream.py',
                 centers=DEFAULT_CENTER, scales=DEFAULT_SCALE,
                 rate_hz=50.0, joint=None,
                 transport='ssh', port=8123):
        """
        Args:
            joints: joint names to mirror (default: all 8, JOINTS).
            host: SSH target (ssh) or TCP host (tcp) running servo_stream.py.
            remote_python: absolute python path on the Pi (ssh only).
            remote_script: servo_stream.py path on the Pi (ssh only).
            centers: servo angle (deg) at each joint's standby pose —
                scalar, {joint: deg} dict, or list aligned with joints.
            scales: servo deg per rad of joint motion — same forms;
                negative for reversed servo mounting.
            rate_hz: max command rate (analog PWM servos: 50 Hz).
            joint: legacy single-joint shorthand for joints=[joint].
            transport: 'ssh' (Pi over SSH) or 'tcp' (ESP32 raw socket).
            port: TCP port for transport='tcp' (default 8123; ssh ignores).
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
        self._transport = transport
        self._sock = None
        self.proc = None
        self._rfile = None
        self._wfile = None

        if transport == 'ssh':
            cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                   # keepalives: detect a dead Pi link in ~15s instead of
                   # hanging on a silently-dropped WiFi connection
                   '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3',
                   host, f'{remote_python} {remote_script}']
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            self._rfile = self.proc.stdout
            self._wfile = self.proc.stdin
        elif transport == 'tcp':
            # Connect with an 8s connect timeout, then drop to blocking
            # I/O (no read timeout) — matches the SSH path's behavior: if
            # the remote dies the write eventually fails rather than
            # silently dropping frames.
            try:
                self._sock = socket.create_connection((host, port), timeout=8)
            except OSError as e:
                raise RuntimeError(
                    f"servo stream failed to connect to {host}:{port}: {e}")
            self._sock.settimeout(None)  # blocking I/O for streaming
            self._rfile = self._sock.makefile('r')
            self._wfile = self._sock.makefile('w')
        else:
            raise ValueError(
                f"unknown transport {transport!r} (expected 'ssh' or 'tcp')")

        ready = self._rfile.readline().strip()
        if ready != f'ready {len(JOINTS)}':
            self._cleanup()
            raise RuntimeError(f"servo stream failed on {host}: {ready!r}")

        # Park every channel at its center and let the servos settle
        # before the loop starts
        self._send_deg(self._park_degs())
        time.sleep(1.0)

    def _cleanup(self):
        """Close whichever transport resources are open (best-effort)."""
        try:
            if self._wfile is not None:
                self._wfile.close()
        except Exception:
            pass
        try:
            if self._rfile is not None:
                self._rfile.close()
        except Exception:
            pass
        if self._transport == 'tcp' and self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._transport == 'ssh' and self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass

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
        self._wfile.write(line + "\n")
        self._wfile.flush()

    def close(self):
        """Park at centers, stop pulsing, close the servo stream."""
        try:
            self._send_deg(self._park_degs())
            time.sleep(0.8)
            self._wfile.write("q\n")
            self._wfile.flush()
        except Exception:
            pass
        if self._transport == 'ssh':
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        else:
            # TCP: wait briefly for the "stopped" reply, then close files
            # and the socket.
            try:
                self._wfile.flush()
                self._rfile.readline()  # expect "stopped"
            except Exception:
                pass
            self._cleanup()
