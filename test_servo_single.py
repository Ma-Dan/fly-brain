#!/usr/bin/env python3
"""单舵机测试工具：指定 PCA9685 通道 + 角度，一次只动一个舵机。

装腿/标定时逐通道核对 servo_mirror 的角度映射。通过 Pi 上已部署的
servo_stream.py（SSH + stdin 8 值协议，同 ServoMirror / test_servo_groups），
内部记录每个通道的当前角度 —— 每条命令只改一个通道的值，其余通道
发送各自当前值（物理上不动），弱电源下任意时刻最多一个舵机在动。

通道 -> 关节（q1lite.xml actuator 顺序，与生产一致）：
    ch0=FR_hip ch1=FR_knee ch2=RR_hip ch3=RR_knee
    ch4=FL_hip ch5=FL_knee ch6=RL_hip ch7=RL_knee
关节角 -> 舵机角直接调用 q1lite_bridge.servo_mirror.map_targets（生产同款）：
    servo_deg = 90 + 57.29578 * (joint_rad - stand_rad)
    FR/RL hip stand=+45°, RR/FL hip stand=-45°, knee stand=0°

注意：
  * 连接时 servo_stream 会先释放全部舵机（上电清零 PWM 保险）—— 已装好的
    腿会失力下垂。这是安装/标定工具，别在整机承重站立时连接。
  * 会话从"所有通道在 90°"起算；退出时把偏离的通道逐个回中 90° 再释放
    （一次一个，间隔 0.8s）。
  * 角度范围 0-180，越界自动截断。

用法：
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_single.py                              # 交互模式
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_single.py --ch 0 --angle 90           # ch0(FR_hip)->90°
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_single.py --ch 0 --joint 45           # FR_hip 关节 45°(=standby)->90°
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_single.py --ch 1 --angle 90 --hold -1  # 保持到回车（装摆臂）
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_single.py --dry-run --ch 0 --angle 120# 只打印换算不连硬件
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_single.py --host ubuntu@192.168.1.141

交互命令：
    <ch> <servo角>   如 `0 120`（舵机角 0-180）
    j <ch> <关节角>  如 `j 0 45`（度，经 mirror 公式换算）
    map              打印 ch/joint/standby/sprawl 映射表
    park             逐通道回中 90°
    q                逐通道回中 -> 释放舵机 -> 退出
"""
import argparse
import math
import subprocess
import sys
import time

import numpy as np

from q1lite_bridge.q1lite_adaptor import HIP0, STAND_POSE
from q1lite_bridge.servo_mirror import (
    DEFAULT_CENTER, DEFAULT_SCALE, JOINTS, map_targets,
)

HOST = 'ubuntu@192.168.1.141'
REMOTE_PY = '/home/ubuntu/miniconda3/envs/lerobot/bin/python'
REMOTE_SCRIPT = '/home/ubuntu/servo_stream.py'
N_SERVOS = 8
PARK_GAP_S = 0.8          # park 时逐通道间隔（让舵机动完再动下一个）


def stand_rad(joint):
    """关节 standby 角（rad）：hip=HIP0，knee=0。"""
    leg, part = joint.split('_')
    return HIP0[leg] if part == 'hip' else 0.0


def servo_from_joint_deg(ch, joint_deg):
    """关节角(°) -> 舵机角(°)。走生产同款 map_targets，与 --servo 完全一致。"""
    targets = np.array(STAND_POSE, dtype=np.float64)
    targets[ch] = math.radians(joint_deg)
    return float(map_targets(targets, JOINTS, DEFAULT_CENTER, DEFAULT_SCALE)[ch])


def joint_from_servo_deg(ch, servo_deg):
    """舵机角(°) -> 关节角(°)：mirror 公式的逆，用于核对。"""
    return math.degrees(stand_rad(JOINTS[ch])
                        + (servo_deg - DEFAULT_CENTER) / DEFAULT_SCALE)


def clamp_deg(deg):
    return max(0.0, min(180.0, float(deg)))


def print_map():
    print(f"\n{'ch':>2}  {'joint':<8}  {'stand(关节°)':>13}  "
          f"{'servo@stand':>11}  {'servo@关节0(sprawl)':>18}")
    for ch, joint in enumerate(JOINTS):
        stand = math.degrees(stand_rad(joint))
        print(f"{ch:>2}  {joint:<8}  {stand:>+13.1f}  "
              f"{servo_from_joint_deg(ch, stand):>11.0f}  "
              f"{servo_from_joint_deg(ch, 0.0):>18.0f}")
    print(f"  换算: servo_deg = {DEFAULT_CENTER:.0f} + {DEFAULT_SCALE:.2f} "
          f"× (joint_rad − stand_rad)\n")


def move(stream, ch, servo_deg):
    """驱动通道 ch 到 servo_deg，打印舵机角/关节角变化。"""
    old = stream.state[ch]
    deg = stream.drive(ch, servo_deg)
    j_old = joint_from_servo_deg(ch, old)
    j_new = joint_from_servo_deg(ch, deg)
    print(f"ch{ch} ({JOINTS[ch]}): servo {old:.0f}° -> {deg:.0f}°  "
          f"| 关节 {j_old:+.1f}° -> {j_new:+.1f}°")
    return deg


class ServoStream:
    """servo_stream.py 连接 + 每通道当前角度记录。dry-run 时不连硬件。"""

    def __init__(self, proc):
        self.proc = proc
        self.state = [DEFAULT_CENTER] * N_SERVOS   # 会话起算：全部在 90°

    def drive(self, ch, deg):
        """只改变通道 ch；其余通道发送各自当前值（物理不动）。"""
        deg = clamp_deg(deg)
        line = [f"{d:.1f}" for d in self.state]
        line[ch] = f"{deg:.1f}"
        self.state[ch] = deg
        if self.proc is not None:
            self.proc.stdin.write(" ".join(line) + "\n")
            self.proc.stdin.flush()
        return deg

    def park(self):
        """偏离 90° 的通道逐个回中，一次只动一个。"""
        for ch in range(N_SERVOS):
            if abs(self.state[ch] - DEFAULT_CENTER) > 0.05:
                move(self, ch, DEFAULT_CENTER)
                if self.proc is not None:
                    time.sleep(PARK_GAP_S)
        print("全部通道已回中 90°。")

    def release(self):
        """发 'q' 让 servo_stream 释放全部舵机（失力，无同时运动）。"""
        if self.proc is None:
            return
        try:
            self.proc.stdin.write("q\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        print("舵机已释放（失力），连接关闭。")


def parse_ch(tok):
    ch = int(tok)
    if not 0 <= ch < N_SERVOS:
        raise ValueError(f"通道号需 0-{N_SERVOS - 1}")
    return ch


def repl(stream):
    print_map()
    print("命令: `<ch> <servo角>` | `j <ch> <关节角>` | map | park | q")
    print("（本会话假定所有通道当前在 90°；若有舵机停在别处，先 `N 90` 归位）")
    while True:
        try:
            raw = input("servo> ").strip()
        except EOFError:
            return
        except KeyboardInterrupt:
            print()
            return
        toks = raw.split()
        if not toks:
            continue
        cmd = toks[0].lower()
        try:
            if cmd == "q":
                return
            if cmd == "map":
                print_map()
            elif cmd == "park":
                stream.park()
            elif cmd == "j":
                ch = parse_ch(toks[1])
                jdeg = float(toks[2])
                servo = servo_from_joint_deg(ch, jdeg)
                move(stream, ch, servo)
                print(f"  请求: {JOINTS[ch]} 关节 {jdeg:+.1f}° -> servo {servo:.1f}°")
            else:
                ch = parse_ch(toks[0])
                move(stream, ch, float(toks[1]))
        except (IndexError, ValueError) as e:
            print(f"输入无效（{e}）。用法: `0 120` / `j 0 45` / map / park / q")


def main():
    parser = argparse.ArgumentParser(
        description='单舵机测试：指定通道+角度，一次只动一个舵机（弱电源安全）')
    parser.add_argument('--host', default=HOST, help='SSH 目标 (默认 %(default)s)')
    parser.add_argument('--remote-python', default=REMOTE_PY)
    parser.add_argument('--remote-script', default=REMOTE_SCRIPT)
    parser.add_argument('--ch', type=int, choices=range(N_SERVOS),
                        help='PCA9685 通道 0-7')
    parser.add_argument('--angle', type=float, help='舵机角 0-180')
    parser.add_argument('--joint', type=float,
                        help='关节角(°)，经 mirror 公式换算成舵机角')
    parser.add_argument('--hold', type=float, default=3.0,
                        help='单次模式保持秒数；-1=保持到按回车 (默认 %(default)s)')
    parser.add_argument('--dry-run', action='store_true',
                        help='不连硬件，仅打印角度换算')
    args = parser.parse_args()

    if (args.angle is not None or args.joint is not None) and args.ch is None:
        parser.error('--angle/--joint 需要配合 --ch')
    if args.angle is not None and args.joint is not None:
        parser.error('--angle 与 --joint 二选一')

    proc = None
    if not args.dry_run:
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3',
               args.host, f'{args.remote_python} {args.remote_script}']
        print(f"连接 {args.host} ...（连接会先释放全部舵机，已装腿会失力）")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        out = proc.stdout
        assert out is not None  # stdout=PIPE -> text stream
        ready = out.readline().strip()
        if not ready.startswith('ready'):
            print(f"连接失败: {ready!r}")
            proc.kill()
            sys.exit(1)
        print(f"servo_stream 就绪（{ready}）")
    else:
        print("DRY-RUN：不连接硬件，仅换算角度。")

    stream = ServoStream(proc)
    try:
        if args.ch is None:
            repl(stream)
        else:
            if args.joint is not None:
                servo = servo_from_joint_deg(args.ch, args.joint)
                print(f"{JOINTS[args.ch]} 关节 {args.joint:+.1f}° -> servo {servo:.1f}°")
            else:
                servo = args.angle if args.angle is not None else DEFAULT_CENTER
            move(stream, args.ch, servo)
            if args.hold < 0:
                input("舵机保持中，装好摆臂后按 Enter 回中并释放 ... ")
            else:
                time.sleep(args.hold)
    finally:
        stream.park()
        stream.release()


if __name__ == '__main__':
    main()
