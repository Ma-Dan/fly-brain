#!/usr/bin/env python3
"""MJPEG HTTP server for the Raspberry Pi USB camera (GEMBIRD USB2.0 PC CAMERA).

Deployed on the Raspberry Pi as /home/ubuntu/camera_stream.py (auto-deployed
by q1lite_bridge/pi_camera.py). Captures /dev/videoN with OpenCV and serves
the frames as a multipart/x-mixed-replace MJPEG stream, so the Mac side can
display the real robot's camera view in place of the MuJoCo virtual camera
(fly_embodied_q1lite.py --camera real).

Prints "ready" on stdout once the camera delivers frames and the HTTP server
is listening. 'q' on stdin, EOF (ssh dropped) or SIGTERM stops the server
cleanly. Mirrors the servo_stream.py lifecycle ("ready"/"q"/"stopped").

Usage (on the Pi):
    /home/ubuntu/miniconda3/envs/lerobot/bin/python /home/ubuntu/camera_stream.py
        --port 8080 --device 0 --width 640 --height 480 --fps 30

Endpoints:
    /stream    multipart MJPEG (for the live display)
    /snapshot  single JPEG image (quick check: curl -o f.jpg http://pi:8080/snapshot)
"""
import argparse
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"


class CameraStream:
    """Single capture thread feeding a latest-frame buffer for N readers."""

    def __init__(self, device, width, height, fps, quality):
        self.quality = int(quality)
        self._jpeg = None
        self._counter = -1
        self._lock = threading.Condition()
        self._stop = threading.Event()

        self.cap = cv2.VideoCapture(int(device))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open /dev/video{device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def wait_first_frame(self, timeout=10.0):
        with self._lock:
            return self._lock.wait_for(
                lambda: self._jpeg is not None or self._stop.is_set(), timeout)

    def _capture_loop(self):
        encode = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                with self._lock:
                    self._stop.set()
                    self._lock.notify_all()
                break
            ok, buf = cv2.imencode('.jpg', frame, encode)
            if not ok:
                continue
            with self._lock:
                self._jpeg = buf.tobytes()
                self._counter += 1
                self._lock.notify_all()

    def latest(self):
        """(counter, jpeg) of the most recent frame (None if none yet)."""
        with self._lock:
            return self._counter, self._jpeg

    def wait_frame(self, after, timeout=5.0):
        """Block until a frame newer than `after` arrives; None on stop/timeout."""
        with self._lock:
            ok = self._lock.wait_for(
                lambda: self._counter > after or self._stop.is_set(), timeout)
            if ok and not self._stop.is_set():
                return self._counter, self._jpeg
            return None

    def stop(self):
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        self._thread.join(timeout=2.0)
        self.cap.release()


def make_handler(stream):
    class Handler(BaseHTTPRequestHandler):
        # MJPEG framing writes small trailing bytes per frame; Nagle +
        # delayed-ACK stalls those ~200ms each, capping throughput at
        # ~5fps. One write per frame + NODELAY restores full rate.
        disable_nagle_algorithm = True

        def do_GET(self):
            if self.path == '/snapshot':
                _, jpeg = stream.latest()
                if jpeg is None:
                    self.send_error(503, 'camera warming up')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path == '/raw':
                # throughput diagnostic: pour the latest frame repeatedly,
                # no Condition waiting — isolates the write path
                self.send_response(200)
                self.end_headers()
                served = 0
                t_start = time.time()
                try:
                    while not stream._stop.is_set():
                        _, jpeg = stream.latest()
                        if jpeg:
                            self.wfile.write(jpeg)
                            self.wfile.flush()
                            served += 1
                            now = time.time()
                            if now - t_start >= 2.0:
                                print(f"raw: {served/(now-t_start):.1f} parts/s",
                                      flush=True)
                                served = 0
                                t_start = now
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path in ('/', '/stream'):
                self.send_response(200)
                self.send_header('Content-Type',
                                 f'multipart/x-mixed-replace; boundary={BOUNDARY}')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                after = -1
                served = 0
                t_wait = 0.0
                t_write = 0.0
                t_start = time.time()
                try:
                    while not stream._stop.is_set():
                        t0 = time.time()
                        got = stream.wait_frame(after)
                        t1 = time.time()
                        if got is None:
                            break
                        after, jpeg = got
                        # single write per frame: headers + jpeg + crlf
                        # (three small writes hit the Nagle/delayed-ACK
                        # 200ms-per-frame stall over WiFi)
                        part = (f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n'
                                f'Content-Length: {len(jpeg)}\r\n\r\n'
                                ).encode() + jpeg + b'\r\n'
                        self.wfile.write(part)
                        self.wfile.flush()
                        t2 = time.time()
                        t_wait += t1 - t0
                        t_write += t2 - t1
                        served += 1
                        now = time.time()
                        if now - t_start >= 2.0:
                            dt = now - t_start
                            print(f"stream: {served/dt:.1f} parts/s "
                                  f"(wait {t_wait/dt*100:.0f}% "
                                  f"write {t_write/dt*100:.0f}%)", flush=True)
                            served = 0
                            t_wait = 0.0
                            t_write = 0.0
                            t_start = now
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client went away
            else:
                self.send_error(404, 'Not Found')

        def log_message(self, format, *args):
            pass  # silence per-request logging

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--device', type=int, default=0,
                    help='/dev/videoN index (default 0)')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--quality', type=int, default=80,
                    help='JPEG quality 1-100 (default 80)')
    args = ap.parse_args()

    try:
        stream = CameraStream(args.device, args.width, args.height,
                               args.fps, args.quality)
    except RuntimeError as e:
        print(f"error: {e}", flush=True)
        sys.exit(1)
    if not stream.wait_first_frame():
        print("error: no frames from camera", flush=True)
        stream.stop()
        sys.exit(1)

    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(stream))
    except OSError as e:
        print(f"error: cannot bind {args.host}:{args.port}: {e}", flush=True)
        stream.stop()
        sys.exit(1)

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"ready :{args.port} /dev/video{args.device} "
          f"{args.width}x{args.height}@{args.fps}", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def watch_stdin():
        try:
            for line in sys.stdin:
                if line.strip() == 'q':
                    stop.set()
                    return
            stop.set()  # EOF (ssh dropped) -> exit, don't orphan the camera
        except Exception:
            stop.set()

    threading.Thread(target=watch_stdin, daemon=True).start()

    # Periodic capture-rate heartbeat (diagnostics: confirms the capture
    # thread keeps up inside this process while clients stream)
    last_count = -1
    last_t = time.time()
    while not stop.wait(timeout=1.0):
        if stream._stop.is_set():       # capture thread died (camera unplugged)
            stop.set()
            continue
        counter, _ = stream.latest()
        now = time.time()
        if now - last_t >= 2.0:
            fps = (counter - last_count) / (now - last_t)
            print(f"capture: {fps:.1f} fps (frame {counter})", flush=True)
            last_count = counter
            last_t = now

    print("stopped", flush=True)
    server.shutdown()
    server.server_close()
    stream.stop()


if __name__ == '__main__':
    main()
