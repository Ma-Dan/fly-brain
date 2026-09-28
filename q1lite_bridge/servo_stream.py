#!/usr/bin/env python3
"""stdin-streamed PWM servo driver (MG90D via pigpio, GPIO 18).

Deployed on the Raspberry Pi as /home/ubuntu/servo_stream.py. Reads one
angle in degrees per line from stdin and drives the servo — no sleeps,
designed for continuous ~50Hz streaming from q1lite_servo_bridge.py.

Pulse mapping matches /home/ubuntu/servo.py: 500-2500 us for 0-180 deg.
'q' or EOF stops pulsing (servo set free) and exits cleanly.

Usage (on the Pi):
    /home/ubuntu/miniconda3/envs/lerobot/bin/python /home/ubuntu/servo_stream.py
    echo 90 | ...   # then stream lines
"""
import sys

import pigpio

SERVO_PIN = 18
PULSE_MIN, PULSE_MAX = 500, 2500  # microseconds (0..180 deg)


def main():
    pi = pigpio.pi()
    if not pi.connected:
        print("error: pigpiod not running", flush=True)
        sys.exit(1)
    print("ready", flush=True)
    try:
        for line in sys.stdin:
            cmd = line.strip()
            if not cmd:
                continue
            if cmd == "q":
                break
            try:
                deg = float(cmd)
            except ValueError:
                continue
            deg = max(0.0, min(180.0, deg))
            pw = PULSE_MIN + (deg / 180.0) * (PULSE_MAX - PULSE_MIN)
            pi.set_servo_pulsewidth(SERVO_PIN, pw)
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, 0)  # stop pulsing -> servo free
        pi.stop()
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
