"""PCA9685 舵机标定/测试工具（8x MG90D，I2C 通道 0-7）。

用途：在树莓派上交互式标定/测试 PCA9685 驱动的 8 个舵机。
部署到 Pi 如 /home/ubuntu/servo_pca9685.py，接线同 servo_stream.py：
    PCA9685 SDA -> Pi pin 3, SCL -> Pi pin 5, VCC -> 3.3V, 共地；
    舵机电源由外部 5V 电源接入 V+（绝不要用 Pi 的 5V —— 8 个舵机电流可达数安培）。

用法（在 Pi 上）：
    /home/ubuntu/miniconda3/envs/lerobot/bin/python /home/ubuntu/servo_pca9685.py
"""
import time

from adafruit_servokit import ServoKit

N_SERVOS = 8
PULSE_MIN, PULSE_MAX = 500, 2500  # 微秒（对应 0~180 度，MG90D）

try:
    kit = ServoKit(channels=16, frequency=50)
    # PCA9685 构造时不会清除之前正在运行的 PWM（只复位 MODE1）—— 上电先清零全部 16 通道
    for ch in kit._pca.channels:
        ch.duty_cycle = 0
    for i in range(N_SERVOS):
        kit.servo[i].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
except Exception as e:
    print(f"错误：无法初始化 PCA9685（{e}）。请确认 I2C 已启用、接线正确"
          f"（i2cdetect -y 1 应看到 0x40）。")
    raise SystemExit(1)

current = 0  # 当前通道 (0-7)


def set_angle(ch, angle):
    # 将角度限制在 0~180 范围内，防止超出机械限位
    angle = max(0, min(180, angle))
    kit.servo[ch].angle = angle
    time.sleep(0.5)  # 等待舵机转动到位
    print(f"通道 {ch} 已转到 {angle} 度")


def set_all(angle):
    # 8 个通道同时转到同一角度
    angle = max(0, min(180, angle))
    for ch in range(N_SERVOS):
        kit.servo[ch].angle = angle
    time.sleep(0.5)  # 等待舵机转动到位
    print(f"8 个通道已同时转到 {angle} 度")


try:
    while True:
        user_input = input(f"[通道 {current}] 请输入角度 (0~180)，"
                           f"c N 切换通道，all 角度 同步 8 通道，q 退出: ").strip()

        if user_input.lower() == 'q':
            print("退出程序。")
            break

        toks = user_input.split()
        if not toks:
            continue

        if toks[0] == 'c':
            if len(toks) < 2:
                print("用法：c N（N 为 0-7 的通道号）")
                continue
            try:
                n = int(toks[1])
            except ValueError:
                print("通道号无效，请输入 0-7 的整数。")
                continue
            if not 0 <= n < N_SERVOS:
                print("通道号超出范围（0-7）。")
                continue
            current = n
            print(f"当前通道切换为 {current}。")
            continue

        if toks[0] == 'all':
            if len(toks) < 2:
                print("用法：all 角度（0~180）")
                continue
            try:
                angle = float(toks[1])
            except ValueError:
                print("输入无效，请输入一个数字。")
                continue
            set_all(angle)
            continue

        try:
            angle = float(toks[0])
        except ValueError:
            print("输入无效，请输入一个数字。")
            continue
        set_angle(current, angle)

finally:
    # 释放所有舵机（angle=None 置零占空比 -> 舵机无力矩），清零全部通道并反初始化
    for i in range(N_SERVOS):
        kit.servo[i].angle = None
    for ch in kit._pca.channels:
        ch.duty_cycle = 0
    kit._pca.deinit()
