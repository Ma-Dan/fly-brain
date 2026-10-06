#!/usr/bin/env python3
"""
PiCameraViewer: real Raspberry Pi USB-camera feed for the brain-body loop.

Two roles, both fed from one SSH-spawned Pi MJPEG server (same lifecycle
idiom as ServoMirror: 'ready' handshake, 'q'/EOF stop):

  1. DISPLAY — the live camera window. The cv2 GUI runs in a SPAWNED
     CHILD PROCESS, not in this process: under mjpython + MLX/Metal the
     in-process OpenCV cocoa window dies seconds after the brain starts
     stepping (cv2.error on every imshow), and even under plain python
     pumping imshow/waitKey from the physics loop starves the reader
     thread's GIL slices. The child owns the window; frames flow through
     an mp queue (JPEG bytes, ~1.2 MB/s) and key events flow back.

  2. VISION — get_frame() returns the latest decoded BGR frame for the
     compound-eye bridge (q1lite_vision.RealCameraVisualBridge), which
     turns the real camera into the fly's eyes (--camera real --visual).

Usage (standalone smoke test):
    python q1lite_bridge/pi_camera.py                      # 10s live feed
    python q1lite_bridge/pi_camera.py --duration 0         # until ESC/q
"""
import argparse
import http.client
import multiprocessing as mp
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

DEFAULT_HOST = 'ubuntu@192.168.1.141'
DEFAULT_PORT = 8080
DEFAULT_REMOTE_PYTHON = '/home/ubuntu/miniconda3/envs/lerobot/bin/python'
DEFAULT_REMOTE_SCRIPT = '/home/ubuntu/camera_stream.py'


def _camera_display_child(frame_q, ctrl_q, window):
    """Child process: owns the cv2 window, isolated from mjpython/MLX.

    Receives JPEG bytes from the parent, decodes + displays them, and
    reports key presses / closure back through ctrl_q as
    ('key', keycode) / ('closed', reason) tuples.
    """
    import cv2  # noqa: F811  (child-local import; already imported anyway)
    gui_failures = 0
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        while True:
            jpeg = frame_q.get()          # blocks; None sentinel = exit
            if jpeg is None:
                break
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                continue
            try:
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                gui_failures = 0
            except cv2.error:
                # intermittent cocoa exceptions: skip frames, give up
                # only when persistent
                gui_failures += 1
                if gui_failures > 60:
                    _put(ctrl_q, ('closed',
                                  'OpenCV GUI failed 60 consecutive frames '
                                  'in display process'))
                    break
                continue
            if key in (27, ord('q')):
                _put(ctrl_q, ('closed', 'ESC/q pressed in camera window'))
                break
            if key != 255 and key < 128:
                _put(ctrl_q, ('key', key))
    except (EOFError, KeyboardInterrupt):
        pass                # parent went away / interrupted
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        _put(ctrl_q, ('closed', 'display process exiting'))


def _put(q, item):
    """Non-blocking put that silently drops when the queue is full."""
    try:
        q.put_nowait(item)
    except Exception:
        pass


class PiCameraViewer:
    """Real-camera viewer + frame source backed by an SSH-spawned Pi
    MJPEG server.

    Lifecycle mirrors ServoMirror: SSH spawns the remote server and waits
    for "ready" on stdout; a daemon thread pulls /stream, parses the
    multipart parts and keeps the latest JPEG + decoded frame; a spawned
    child process owns the cv2 display window. is_running() goes False
    when the SSH server or the display child dies, or close() is called.
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 remote_python=DEFAULT_REMOTE_PYTHON,
                 remote_script=DEFAULT_REMOTE_SCRIPT,
                 device=0, width=640, height=480, fps=30.0, quality=80,
                 window='Q1 Lite — Real Camera (Pi)', display=True):
        self.host = host
        self.port = int(port)
        self.window = window
        self._ip = host.split('@')[-1]      # strip user@ for plain HTTP
        self._jpeg = None
        self._frame = None
        self._frame_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error = None
        self._close_reason = None
        self._last_sent = 0.0

        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
               # keepalives: detect a dead Pi link in ~15s instead of
               # hanging on a silently-dropped WiFi connection
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3',
               host,
               f'{remote_python} {remote_script}'
               f' --port {self.port} --device {int(device)}'
               f' --width {int(width)} --height {int(height)}'
               f' --fps {float(fps)} --quality {int(quality)}']
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        stdout = self.proc.stdout
        stdin = self.proc.stdin
        assert stdout is not None and stdin is not None
        self._stdin = stdin
        self._stdout = stdout
        ready = stdout.readline().strip()
        if not ready.startswith('ready'):
            self.proc.kill()
            tail = stdout.read()
            raise RuntimeError(
                f"camera stream failed on {host}: {ready!r} {tail}")

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        # Drain SSH stdout (server heartbeat prints): if nobody reads it,
        # the 64KB pipe buffer eventually fills and the server blocks on
        # print(). Keeps the last few lines so a silent SSH death can be
        # explained (capture-thread death, Pi reboot, WiFi drop...).
        self._server_tail = deque(maxlen=5)
        threading.Thread(target=self._drain_stdout, daemon=True).start()

        # Display child process (skip for headless vision-only use)
        self._display = None
        if display:
            ctx = mp.get_context('spawn')
            self._frame_q = ctx.Queue(maxsize=2)
            self._ctrl_q = ctx.Queue()
            self._display = ctx.Process(
                target=_camera_display_child,
                args=(self._frame_q, self._ctrl_q, window),
                daemon=True)
            self._display.start()

    # ── MJPEG reader (background) ────────────────────────────────────

    def _reader_loop(self):
        """Pull /stream forever; keep the latest JPEG + decoded frame."""
        while not self._stop.is_set():
            conn = None
            try:
                conn = http.client.HTTPConnection(self._ip, self.port,
                                                  timeout=10)
                conn.request('GET', '/stream')
                resp = conn.getresponse()
                if resp.status != 200:
                    self._error = f'stream HTTP {resp.status}'
                    time.sleep(1.0)
                    continue
                buf = b''
                while not self._stop.is_set():
                    # part headers end with \r\n\r\n
                    while b'\r\n\r\n' not in buf:
                        chunk = resp.read(4096)
                        if not chunk:
                            raise ConnectionError('stream ended')
                        buf += chunk
                    headers, buf = buf.split(b'\r\n\r\n', 1)
                    m = re.search(rb'Content-Length:\s*(\d+)', headers,
                                  re.IGNORECASE)
                    if m is None:
                        continue          # not a JPEG part; resync
                    n = int(m.group(1))
                    while len(buf) < n:
                        chunk = resp.read(min(65536, n - len(buf)))
                        if not chunk:
                            raise ConnectionError('stream ended mid-frame')
                        buf += chunk
                    jpeg, buf = buf[:n], buf[n:]
                    if buf.startswith(b'\r\n'):
                        buf = buf[2:]
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                         cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._lock:
                            self._jpeg = jpeg
                            self._frame = frame
                            self._frame_count += 1
            except Exception as e:
                if not self._stop.is_set():
                    self._error = str(e)
                    time.sleep(1.0)      # auto-reconnect on dropouts
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

    def _drain_stdout(self):
        try:
            for line in self._stdout:
                line = line.rstrip()
                if line:
                    self._server_tail.append(line)
        except Exception:
            pass

    # ── Viewer API (mirrors the MuJoCo passive viewer subset) ────────

    def is_running(self):
        """False once the SSH server or display child died, or close()."""
        if self._stop.is_set():
            return False
        if self.proc.poll() is not None:
            return False
        if self._display is not None and not self._display.is_alive():
            return False
        return True

    def get_frame(self):
        """Latest BGR frame (HxWx3 uint8) or None while warming up.

        Returns the reader's array without copying — imdecode creates a
        fresh array per frame and the old one is never mutated, so the
        reference stays valid even after newer frames arrive.
        """
        with self._lock:
            return self._frame

    @property
    def frame_count(self):
        with self._lock:
            return self._frame_count

    @property
    def last_error(self):
        return self._error

    @property
    def close_reason(self):
        """Why the viewer last signalled closed (None if it hasn't)."""
        return self._close_reason or 'unknown'

    def explain_exit(self):
        """Full human-readable explanation for why the stream ended.

        Call after is_running() returned False — reports the SSH exit
        status, the Pi server's last output lines, and the display
        child's close reason, so silent deaths (Pi reboot, WiFi drop,
        camera read failure, GUI death) are diagnosable.
        """
        parts = []
        code = self.proc.poll()
        if code is not None:
            tail = ' | '.join(self._server_tail) or '(no server output)'
            parts.append(f'ssh exited with code {code}; server said: {tail}')
        if self._display is not None and not self._display.is_alive():
            parts.append(f'display: {self._close_reason or "child died"}')
        if self._error:
            parts.append(f'reader: {self._error}')
        return '; '.join(parts) or 'closed by user'

    def show(self, on_key=None):
        """Feed the display child + poll its key/close events.

        Returns True while the viewer is open. Internally rate-limited
        to ~30 sends/s (camera rate); JPEG bytes go to the child, key
        events come back and are routed to on_key(keycode) — the same
        callback contract as the MuJoCo viewer (1=sugar, 3=lc4,
        SPACE=auto-demo, ...). ESC/q or closing the child window makes
        this return False.
        """
        if self._display is None:
            return True                    # headless vision-only mode
        now = time.monotonic()
        if now - self._last_sent >= 1.0 / 30.0:
            self._last_sent = now
            with self._lock:
                jpeg = self._jpeg
            if jpeg is not None:
                try:
                    self._frame_q.put_nowait(jpeg)
                except queue.Full:
                    # drop the stale frame, keep the pipeline fresh
                    try:
                        self._frame_q.get_nowait()
                        self._frame_q.put_nowait(jpeg)
                    except (queue.Empty, queue.Full):
                        pass
        # drain key/close events from the child
        while True:
            try:
                kind, val = self._ctrl_q.get_nowait()
            except queue.Empty:
                break
            if kind == 'closed':
                self._close_reason = val
                return False
            if kind == 'key' and on_key is not None:
                on_key(val)
        return True

    # ── Shutdown ─────────────────────────────────────────────────────

    def _stop_remote(self):
        try:
            self._stdin.write("q\n")
            self._stdin.flush()
            self._stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def close(self):
        """Stop the display child, park the Pi server, release the stream."""
        self._stop.set()
        if self._display is not None:
            try:
                self._frame_q.put_nowait(None)     # sentinel: exit loop
            except Exception:
                pass
            self._display.join(timeout=3.0)
            if self._display.is_alive():
                self._display.terminate()
        self._stop_remote()


def _main():
    ap = argparse.ArgumentParser(description='Pi camera viewer smoke test')
    ap.add_argument('--host', default=DEFAULT_HOST)
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--quality', type=int, default=80)
    ap.add_argument('--duration', type=float, default=10.0,
                    help='seconds to run (0 = until ESC/q, default 10)')
    args = ap.parse_args()

    print(f"Starting Pi camera stream on {args.host}:{args.port} ...")
    try:
        cam = PiCameraViewer(
            host=args.host, port=args.port, device=args.device,
            width=args.width, height=args.height,
            fps=args.fps, quality=args.quality,
            window='Pi Camera (ESC/q to quit)')
    except RuntimeError as e:
        print(f"Failed: {e}")
        sys.exit(1)
    print(f"Streaming {args.width}x{args.height}@{args.fps} from {args.host} "
          f"— ESC/q to quit"
          + (f", {args.duration:.0f}s" if args.duration > 0 else "")
          + ".")

    t0 = time.time()
    try:
        while cam.is_running():
            if not cam.show():
                break
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - t0
        cam.close()
        print(f"Done — {cam.frame_count} frames received in {elapsed:.1f}s "
              f"({cam.frame_count/max(elapsed,1e-6):.1f} fps)."
              + (f" last error: {cam.last_error}" if cam.last_error else ""))


if __name__ == '__main__':
    _main()
