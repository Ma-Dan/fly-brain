#!/usr/bin/env python3
"""
Q1 Lite single-servo bring-up: mirror one MuJoCo joint onto a real PWM
servo (MG90D) attached to a Raspberry Pi on the LAN.

Architecture matches the MG90S servo model in q1lite_bridge: the
controller only SENDS position commands at 50Hz — no joint readback.
The CPG's commanded angle for one joint is streamed over a single SSH
connection to servo_stream.py on the Pi, which drives the servo via
pigpio. The simulation runs paced to wall-clock time so the servo
visibly follows the gait in real time.

Angle mapping: servo_deg = CENTER + SCALE * (cmd_rad - STAND_rad) —
the joint's standby pose maps to servo CENTER (default 90 deg,
mechanical mid-range) and joint swings map 1:1 in degrees.

Usage:
    python q1lite_servo_bridge.py                          # FR_hip, 10s gait
    python q1lite_servo_bridge.py --joint FR_knee --duration 20
    python q1lite_servo_bridge.py --drive 0.5              # gentler gait
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from q1lite_bridge import Q1LiteSim, QuadCPG
from q1lite_bridge.q1lite_adaptor import HIP0, LEGS

JOINTS = [f'{leg}_{part}' for leg in LEGS for part in ('hip', 'knee')]
SCENE = str(Path(__file__).resolve().parent / 'q1lite' / 'scene.xml')
DT = 0.001  # sim timestep (s), real-time paced


def joint_index(name):
    leg, part = name.split('_')
    return LEGS.index(leg) * 2 + (0 if part == 'hip' else 1)


def stand_rad(name):
    leg, part = name.split('_')
    return HIP0[leg] if part == 'hip' else 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Q1 Lite single-servo bring-up (MuJoCo -> real MG90D)')
    parser.add_argument('--host', default='ubuntu@192.168.1.141')
    parser.add_argument('--remote-python',
                        default='/home/ubuntu/miniconda3/envs/lerobot/bin/python')
    parser.add_argument('--remote-script', default='/home/ubuntu/servo_stream.py')
    parser.add_argument('--joint', default='FR_hip', choices=JOINTS,
                        help='which simulated joint to mirror (default FR_hip)')
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
    args = parser.parse_args()

    j = joint_index(args.joint)
    stand = stand_rad(args.joint)

    # ── Open the SSH servo stream ────────────────────────────────────
    cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
           args.host, f'{args.remote_python} {args.remote_script}']
    print(f"Connecting to servo stream on {args.host} ...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    ready = proc.stdout.readline().strip()
    if ready != 'ready':
        print(f"Servo stream failed: {ready!r}")
        proc.kill()
        sys.exit(1)
    print("Servo stream ready (pigpio connected).")

    def send(deg):
        proc.stdin.write(f"{max(0.0, min(180.0, deg)):.1f}\n")
        proc.stdin.flush()

    # ── Sim + CPG ────────────────────────────────────────────────────
    sim = Q1LiteSim(SCENE, timestep=DT)
    sim.reset()
    cpg = QuadCPG(dt=DT)

    steps_per_cmd = max(1, int(round(1.0 / (args.rate * DT))))  # 20 @ 50Hz
    interval = steps_per_cmd * DT                               # 0.02 s
    n_steps = int(args.duration / DT)

    print(f"Mirroring {args.joint} (stand {np.degrees(stand):.0f} deg) -> "
          f"servo center {args.center:.0f} deg, {args.rate:.0f}Hz stream, "
          f"drive={args.drive}, turn={args.turn}, {args.duration:.0f}s")
    print("Centering servo, 1s settle ...")
    send(args.center)
    time.sleep(1.0)

    try:
        next_send = time.time()
        t0 = next_send
        last_deg = args.center
        for step in range(n_steps):
            targets = cpg.step(args.drive, args.turn)
            sim.step(targets)

            if step % steps_per_cmd == 0:
                now = time.time()
                if now < next_send:
                    time.sleep(next_send - now)
                next_send += interval
                deg = args.center + args.scale * (targets[j] - stand)
                send(deg)
                last_deg = deg

            if step % 1000 == 0 and step > 0:
                print(f"  [t={step * DT:.1f}s] {args.joint} "
                      f"cmd={np.degrees(targets[j]):+.1f} deg "
                      f"-> servo {last_deg:.0f} deg")

        # Park at center and stop the stream
        send(args.center)
        time.sleep(1.0)
        proc.stdin.write("q\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(timeout=10)
        print("Done — servo parked at center, stream stopped.")
    except KeyboardInterrupt:
        print("\nInterrupted — parking servo.")
    finally:
        try:
            proc.stdin.write("q\n")
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == '__main__':
    main()
