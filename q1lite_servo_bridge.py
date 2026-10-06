#!/usr/bin/env python3
"""
Q1 Lite servo bring-up: mirror MuJoCo joint targets onto real PWM servos
(MG90D x8) driven by a PCA9685 on a Raspberry Pi on the LAN.

Architecture matches the MG90S servo model in q1lite_bridge: the
controller only SENDS position commands at 50Hz — no joint readback.
The CPG's commanded angles for the mirrored joints are streamed over a
single SSH connection (ServoMirror) to servo_stream.py on the Pi, which
drives the PCA9685 over I2C (channels 0-7, actuator order). The
simulation runs paced to wall-clock time so the servos visibly follow
the gait in real time.

Angle mapping: servo_deg = CENTER + SCALE * (cmd_rad - STAND_rad) —
the joint's standby pose maps to servo CENTER (default 90 deg,
mechanical mid-range) and joint swings map 1:1 in degrees. Default is
ALL 8 joints (PCA9685 channels 0-7); --joint selects one for
single-joint bring-up, --map gives per-joint center/scale overrides
(negative scale for reversed servo mounts).

Usage:
    python q1lite_servo_bridge.py                          # all 8 joints, 10s gait
    python q1lite_servo_bridge.py --joint FR_knee --duration 20
    python q1lite_servo_bridge.py --drive 0.5              # gentler gait
    python q1lite_servo_bridge.py --map '{"FL_hip": {"center": 90, "scale": -57.3}}'
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from q1lite_bridge import Q1LiteSim, QuadCPG
from q1lite_bridge.q1lite_adaptor import HIP0
from q1lite_bridge.servo_mirror import JOINTS, ServoMirror, map_targets

SCENE = str(Path(__file__).resolve().parent / 'q1lite' / 'scene.xml')
DT = 0.001  # sim timestep (s), real-time paced


def stand_rad(name):
    leg, part = name.split('_')
    return HIP0[leg] if part == 'hip' else 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Q1 Lite servo bring-up (MuJoCo -> real MG90D x8 via PCA9685)')
    parser.add_argument('--host', default='ubuntu@192.168.1.141')
    parser.add_argument('--remote-python',
                        default='/home/ubuntu/miniconda3/envs/lerobot/bin/python')
    parser.add_argument('--remote-script', default='/home/ubuntu/servo_stream.py')
    parser.add_argument('--joint', default=None, choices=JOINTS,
                        help='mirror only this joint (default: all 8 joints)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='gait duration in seconds (default 10)')
    parser.add_argument('--drive', type=float, default=1.0,
                        help='CPG forward drive 0..1 (default 1.0)')
    parser.add_argument('--turn', type=float, default=0.0,
                        help='CPG turn drive -1..1 (default 0)')
    parser.add_argument('--center', type=float, default=90.0,
                        help='servo angle (deg) at the joint standby pose')
    parser.add_argument('--scale', type=float, default=57.29578,
                        help='servo deg per rad of joint motion (default 1:1)')
    parser.add_argument('--rate', type=float, default=50.0,
                        help='command stream rate in Hz (default 50)')
    parser.add_argument('--map', default=None,
                        help='JSON per-joint overrides, e.g. '
                             '\'{"FL_hip": {"center": 90, "scale": -57.3}}\'')
    args = parser.parse_args()

    # ── Build per-joint center/scale (complete dicts over the mirrored set) ──
    joints_list = None if args.joint is None else [args.joint]
    all_mirrored = JOINTS if joints_list is None else joints_list
    centers = {j: args.center for j in all_mirrored}
    scales = {j: args.scale for j in all_mirrored}
    if args.map:
        for j, ov in json.loads(args.map).items():
            centers[j] = float(ov.get('center', args.center))
            scales[j] = float(ov.get('scale', args.scale))

    # ── Open the SSH servo stream (ServoMirror handles handshake + park) ──
    print(f"Connecting to servo stream on {args.host} ...")
    try:
        servo = ServoMirror(joints=joints_list, host=args.host,
                            remote_python=args.remote_python,
                            remote_script=args.remote_script,
                            centers=centers, scales=scales, rate_hz=args.rate)
    except (RuntimeError, ValueError) as e:
        print(f"Servo stream failed: {e}")
        sys.exit(1)
    print("Servo stream ready (PCA9685 connected); servos parked at centers, "
          "1s settle ...")

    # ── Sim + CPG ────────────────────────────────────────────────────
    sim = Q1LiteSim(SCENE, timestep=DT)
    sim.reset()
    cpg = QuadCPG(dt=DT)

    steps_per_cmd = max(1, int(round(1.0 / (args.rate * DT))))  # 20 @ 50Hz
    interval = steps_per_cmd * DT                               # 0.02 s
    n_steps = int(args.duration / DT)

    if args.joint is None:
        print(f"Mirroring all 8 joints -> PCA9685 ch0-7 @ {args.host}, "
              f"center {args.center:.0f} deg, scale {args.scale:.1f} deg/rad, "
              f"{args.rate:.0f}Hz stream, drive={args.drive}, turn={args.turn}, "
              f"{args.duration:.0f}s")
        if args.map:
            print(f"  per-joint overrides: {args.map}")
    else:
        stand = stand_rad(args.joint)
        print(f"Mirroring {args.joint} (stand {np.degrees(stand):.0f} deg) -> "
              f"servo center {centers[args.joint]:.0f} deg, scale "
              f"{scales[args.joint]:.1f} deg/rad, {args.rate:.0f}Hz stream, "
              f"drive={args.drive}, turn={args.turn}, {args.duration:.0f}s")

    try:
        next_send = time.time()
        for step in range(n_steps):
            targets = cpg.step(args.drive, args.turn)
            sim.step(targets)

            if step % steps_per_cmd == 0:
                now = time.time()
                if now < next_send:
                    time.sleep(next_send - now)  # keep the sim real-time
                next_send += interval
                servo.send(targets)  # rate-limited to 50Hz internally

            if step % 1000 == 0 and step > 0:
                degs = map_targets(targets, all_mirrored, centers, scales)
                if args.joint is None:
                    print(f"  [t={step * DT:.1f}s] servo deg: {np.round(degs, 0)} "
                          f"(drive={args.drive:.2f}, turn={args.turn:.2f})")
                else:
                    j = JOINTS.index(args.joint)
                    print(f"  [t={step * DT:.1f}s] {args.joint} "
                          f"cmd={np.degrees(targets[j]):+.1f} deg "
                          f"-> servo {degs[j]:.0f} deg")

        # Park at centers and stop the stream
        servo.close()
        print("Done — servos parked at centers, stream stopped.")
    except KeyboardInterrupt:
        print("\nInterrupted — parking servos.")
    finally:
        servo.close()  # idempotent: parks + 'q' + wait, safe if already closed


if __name__ == '__main__':
    main()
