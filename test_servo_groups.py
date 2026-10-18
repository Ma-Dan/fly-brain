#!/usr/bin/env python3
"""PCA9685 舵机分组测试：分组交替摆动，组间释放停顿，排查电源极限。

默认 4 组 × 2 通道（ch0+1, ch2+3, ch4+5, ch6+7），可自定义组数和
每组通道数。通过 servo_stream.py 协议驱动，不依赖 Pi 端临时脚本。
支持两种 transport：ssh（Pi，subprocess）或 tcp（ESP32，raw socket），
协议相同（ready 8 -> 角度行 -> q -> stopped）。

用法:
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_groups.py
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_groups.py --groups 2 --per-group 4
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_groups.py --duration 8 --interval 2
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_groups.py --host ubuntu@192.168.1.141
    /Users/dan/miniconda3/envs/fly/bin/python test_servo_groups.py --transport tcp --host 192.168.1.50
"""
import argparse
import socket
import subprocess
import sys
import time

HOST = 'ubuntu@192.168.1.141'
REMOTE_PY = '/home/ubuntu/miniconda3/envs/lerobot/bin/python'
REMOTE_SCRIPT = '/home/ubuntu/servo_stream.py'
N_SERVOS = 8


def main():
    parser = argparse.ArgumentParser(
        description='PCA9685 舵机分组测试（分组交替摆动，组间释放）')
    parser.add_argument('--host', default=HOST,
                        help='SSH 目标 (默认 %(default)s)')
    parser.add_argument('--transport', choices=['ssh', 'tcp'], default='ssh',
                        help='servo link transport: ssh (Pi) or tcp (ESP32)')
    parser.add_argument('--port', type=int, default=8123,
                        help='TCP port for tcp transport (default 8123)')
    parser.add_argument('--remote-python', default=REMOTE_PY)
    parser.add_argument('--remote-script', default=REMOTE_SCRIPT)
    parser.add_argument('--groups', type=int, default=4,
                        help='分组数 (默认 4)')
    parser.add_argument('--per-group', type=int, default=2,
                        help='每组通道数 (默认 2)')
    parser.add_argument('--duration', type=float, default=6.0,
                        help='每组摆动时长秒 (默认 6)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='组间释放停顿秒 (默认 1)')
    args = parser.parse_args()

    total = args.groups * args.per_group
    if total > N_SERVOS:
        print(f"错误：{args.groups} 组 × {args.per_group} = {total} "
              f"超过 {N_SERVOS} 个通道")
        sys.exit(1)

    # 生成分组：[[0,1], [2,3], [4,5], [6,7]]
    groups = [list(range(i, i + args.per_group))
              for i in range(0, total, args.per_group)]

    # ── 连接 servo_stream ──────────────────────────────────────────
    link = (f"{args.host}:{args.port} (tcp)" if args.transport == 'tcp'
            else f"{args.host} (ssh)")
    print(f"连接 {link} ...")
    if args.transport == 'ssh':
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3',
               args.host, f'{args.remote_python} {args.remote_script}']
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        rfile, wfile = proc.stdout, proc.stdin
        sock = None
    else:
        try:
            sock = socket.create_connection((args.host, args.port), timeout=8)
        except OSError as e:
            print(f"连接失败: {e}")
            sys.exit(1)
        sock.settimeout(None)
        # 关 Nagle：50Hz 小包立即发出，避免攒包加剧突发（与 ServoMirror 一致）
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        rfile, wfile = sock.makefile('r'), sock.makefile('w')
        proc = None

    ready = rfile.readline().strip()
    if not ready.startswith('ready'):
        print(f"连接失败: {ready!r}")
        if proc is not None:
            proc.kill()
        if sock is not None:
            sock.close()
        sys.exit(1)
    print(f"servo_stream 就绪 ({ready})")

    def send(degs):
        wfile.write(" ".join(f"{d:.1f}" for d in degs) + "\n")
        wfile.flush()

    park = [90.0] * N_SERVOS

    try:
        for idx, group in enumerate(groups):
            chs = " ".join(f"ch{c}" for c in group)
            print(f"\n=== 第 {idx + 1}/{args.groups} 组: {chs} ===")
            send(park)
            time.sleep(0.5)

            toggle = False
            t0 = time.time()
            while time.time() - t0 < args.duration:
                degs = list(park)
                for c in group:
                    degs[c] = 0.0 if toggle else 180.0
                send(degs)
                states = "  ".join(f"ch{c}={degs[c]:.0f}" for c in group)
                print(f"  [{time.time() - t0:.1f}s] {states}  (其余=90)",
                      flush=True)
                toggle = not toggle
                time.sleep(1.0)

            # 回中停顿（servo_stream 没有"释放单通道"指令，park 在 90°）
            send(park)
            print(f"  {chs} 回中，停顿 {args.interval:.0f}s ...", flush=True)
            time.sleep(args.interval)

        print("\n全部分组测试完成。")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        try:
            send(park)
            time.sleep(0.5)
            wfile.write("q\n")
            wfile.flush()
        except Exception:
            pass
        if proc is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        if sock is not None:
            try:
                rfile.readline()  # expect "stopped"
            except Exception:
                pass
            try:
                wfile.close()
                rfile.close()
            except Exception:
                pass
            sock.close()
        print("舵机已释放，连接关闭。")


if __name__ == '__main__':
    main()
