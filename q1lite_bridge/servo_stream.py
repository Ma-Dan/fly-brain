#!/usr/bin/env python3
"""stdin-streamed PCA9685 servo driver (8x MG90D via I2C, channels 0-7).

Deployed on the Raspberry Pi as /home/ubuntu/servo_stream.py. Reads one
line of 8 space-separated angles in degrees from stdin and drives the 8
servos — no sleeps, designed for continuous ~50Hz streaming from
q1lite_servo_bridge.py / fly_embodied_q1lite.py --servo.

Channel order = q1lite.xml actuator order (ch0=FR_hip .. ch7=RL_knee):
    FR_hip FR_knee RR_hip RR_knee FL_hip FL_knee RL_hip RL_knee

Pulse mapping for MG90D: 500-2500 us for 0-180 deg (set per channel via
set_pulse_width_range; the ServoKit default 750-2250 us is wrong for
MG90D). Angles are clamped to [0, 180] before writing — values outside
that range RAISE ValueError in ServoKit. Lines with a wrong token count
or unparseable values are silently skipped (robust mid-stream). 'q' or
EOF releases all 8 servos (limp), zeros all 16 PCA9685 channels and
exits cleanly.

Pi setup (once):
    sudo raspi-config nonint do_i2c 0        # enable I2C (or interactive: Interface Options -> I2C)
    sudo apt-get install -y i2c-tools
    sudo usermod -aG i2c ubuntu              # log out/in after this
    i2cdetect -y 1                           # expect 0x40 (default address)
    /home/ubuntu/miniconda3/envs/lerobot/bin/python -m pip install adafruit-circuitpython-servokit

Wiring: PCA9685 SDA -> Pi pin 3, SCL -> Pi pin 5, VCC -> 3.3V, common
GND. Servo power comes from an EXTERNAL 5V supply on V+ — NEVER the Pi
5V rail (8 servos can draw amps and brown out the Pi).

Usage (on the Pi):
    /home/ubuntu/miniconda3/envs/lerobot/bin/python /home/ubuntu/servo_stream.py
    echo "90 90 90 90 90 90 90 90" | ...     # then stream lines
"""
import sys

from adafruit_servokit import ServoKit

N_SERVOS = 8
PULSE_MIN, PULSE_MAX = 500, 2500  # microseconds (0..180 deg, MG90D)


def main():
    try:
        kit = ServoKit(channels=16, frequency=50)
        # the PCA9685 constructor does NOT clear previously-running PWM
        # (only resets MODE1) — zero all 16 channels as startup insurance
        for ch in kit._pca.channels:
            ch.duty_cycle = 0
        for i in range(N_SERVOS):
            kit.servo[i].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
    except Exception as e:
        print(f"error: {e}", flush=True)
        sys.exit(1)
    print(f"ready {N_SERVOS}", flush=True)
    try:
        for line in sys.stdin:
            cmd = line.strip()
            if not cmd:
                continue
            if cmd == "q":
                break
            toks = cmd.split()
            if len(toks) != N_SERVOS:
                continue
            try:
                degs = [float(t) for t in toks]
            except ValueError:
                continue
            for i, deg in enumerate(degs):
                kit.servo[i].angle = max(0.0, min(180.0, deg))
    finally:
        for i in range(N_SERVOS):
            kit.servo[i].angle = None  # release -> servo limp
        for ch in kit._pca.channels:
            ch.duty_cycle = 0
        kit._pca.deinit()
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
