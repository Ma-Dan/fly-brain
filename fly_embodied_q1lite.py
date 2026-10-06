#!/usr/bin/env python3
"""
Embodied Drosophila → Q1 Lite: Brain-body closed loop.

Replaces the NeuroMechFly body with a Q1 Lite 8-DOF spider quadruped.
Q1 Lite is an 8-DOF spider quadruped (hip=yaw, knee=lift), forward = body +Y;
same 138,639-neuron brain, different body.

Usage:
    mjpython fly_embodied_q1lite.py                        # Auto-demo (MuJoCo viewer + virtual eyes)
    mjpython fly_embodied_q1lite.py --stimulus p9          # Manual stimulus
    mjpython fly_embodied_q1lite.py --no-auto              # Keyboard only
    mjpython fly_embodied_q1lite.py --olfactory --gustatory --somatosensory
    mjpython fly_embodied_q1lite.py --servo               # + mirror all 8 joints to real servos
    mjpython fly_embodied_q1lite.py --camera real --visual # Pi camera as compound eyes + live view
    mjpython fly_embodied_q1lite.py --camera real --visual --servo   # full real-robot loop

Camera modes (--camera):
    virtual (default)  MuJoCo offscreen render → compound eyes; MuJoCo
                       viewer shows the sim. Run under mjpython (macOS).
    real               Pi USB camera → compound eyes (real world vision);
                       camera window shows the live feed. The MuJoCo
                       viewer ALSO opens (sim state) so you see both.
                       Works under both mjpython and plain python; the
                       camera display runs in a spawned child process so
                       it never conflicts with MLX/Metal or mjpython.

Keys (work in BOTH the MuJoCo viewer and the camera window):
    1 = Sugar GRNs      -> forward approach
    2 = P9 direct       -> forward walking
    3 = LC4 looming     -> escape
    4 = JO touch        -> stand still
    5 = Bitter GRNs     -> aversion
    6 = Or56a olfactory -> repulsion
    0 = No stimulus     -> spontaneous activity
    SPACE = Toggle auto-demo on/off
"""

import sys
import argparse
import json
import numpy as np
import mujoco
import multiprocessing as mp
import time
from pathlib import Path

from brain_body_bridge import (
    BrainEngine, DNRateDecoder, STIMULI,
)
from q1lite_bridge.q1lite_sim import Q1LiteSim
from q1lite_bridge.q1lite_adaptor import HIP0, Q1LiteAdaptor, QuadCPG
from q1lite_bridge.q1lite_vision import Q1LiteVisualBridge
from q1lite_bridge.servo_mirror import ServoMirror, JOINTS

from visual_system import VisualSystem
from somatosensory import SomatosensorySystem, VibrationSource
from gustatory import GustatorySystem, TasteZone
from olfactory import OlfactorySystem, OdorSource
from brain_monitor import BrainMonitorProcess

try:
    from consciousness import ConsciousnessDetector
except ImportError:
    ConsciousnessDetector = None

# Q1 Lite model path
_Q1LITE_SCENE = Path('./q1lite/scene.xml').resolve()

# ============================================================================
# Auto-demo sequence
# ============================================================================

AUTO_DEMO_SEQUENCE = [
    ('p9',     4.0, 'Forward walking (P9)'),
    ('lc4',    2.0, 'ESCAPE! (LC4 looming)'),
    (None,     2.0, 'Recovery (no stimulus)'),
    ('sugar',  4.0, 'Sugar detected (approach)'),
    (None,     1.5, 'Pause'),
    ('jo',     2.0, 'JO touch (stand still)'),
    (None,     2.0, 'Recovery'),
    ('p9',     3.0, 'Walking again (P9)'),
    ('bitter', 3.0, 'Bitter taste (aversion)'),
    (None,     2.0, 'Pause'),
    ('or56a',  3.0, 'Bad smell (repulsion)'),
    (None,     1.5, 'Recovery'),
]

# ============================================================================
# Sensory Adaptors — convert Q1 Lite sensor data to fly-compatible formats
# ============================================================================

# NOTE: contact-force feedback is intentionally absent — the v1 hardware
# target (SC09 serial bus servos) has no force/torque sensing, so the
# JO touch pathway and bridge-level tactile escape are not wired. Only
# geometric feedback (foot positions) is converted below.

def q1lite_feet_to_fly_end_effectors(q1lite_foot_positions, sim_position):
    """
    Convert Q1 Lite (4, 3) foot positions (m) to fly-compatible (6, 3) array (mm).

    Q1 Lite feet: FL, FR, RL, RR (in m)
    Fly expects: 6 end effectors in mm
    We map: LF→FL, LM→(FL+RL)/2, LH→RL, RF→FR, RM→(FR+RR)/2, RH→RR
    """
    # Convert m to mm
    feet_mm = q1lite_foot_positions * 1000.0
    fly_pos_mm = sim_position * 1000.0

    fl = feet_mm[0]  # FL
    fr = feet_mm[1]  # FR
    rl = feet_mm[2]  # RL
    rr = feet_mm[3]  # RR

    out = np.zeros((6, 3), dtype=np.float64)
    out[0] = fl                    # LF → FL
    out[1] = (fl + rl) / 2.0      # LM → average of FL and RL
    out[2] = rl                    # LH → RL
    out[3] = fr                    # RF → FR
    out[4] = (fr + rr) / 2.0      # RM → average of FR and RR
    out[5] = rr                    # RH → RR

    return out


# ============================================================================
# Main Simulation
# ============================================================================

def main():
    # Lazy import: the spawn'd brain-monitor/camera-display children
    # re-import this module, and a module-level cv2 import here would drag
    # cv2's bundled SDL2 into the children alongside pygame's SDL2 — the
    # objc class-collision warnings and "mysterious crashes" macOS warns
    # about. Importing inside main() keeps cv2 (and SDL2) in this process
    # (and the camera-display child) only.
    from q1lite_bridge.pi_camera import (
        PiCameraViewer,
        DEFAULT_HOST as PI_CAM_HOST,
        DEFAULT_PORT as PI_CAM_PORT,
    )
    from q1lite_bridge.q1lite_vision import RealCameraVisualBridge

    parser = argparse.ArgumentParser(description='Embodied Drosophila → Q1 Lite')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Run headless (no viewer)')
    parser.add_argument('--no-brain', action='store_true',
                        help='Body only — CPG walking, no neural sim')
    parser.add_argument('--no-auto', action='store_true',
                        help='Disable auto-demo (keyboard only)')
    parser.add_argument('--stimulus', type=str, default=None,
                        choices=list(STIMULI.keys()),
                        help='Initial stimulus to activate')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='Max sim duration in seconds (0=unlimited)')
    parser.add_argument('--monitor', action='store_true',
                        help='Open brain monitor window')
    parser.add_argument('--somatosensory', action='store_true',
                        help='Enable vibration/sound via JO neurons (no contact-force sensing)')
    parser.add_argument('--gustatory', action='store_true',
                        help='Enable taste zones (sugar/bitter)')
    parser.add_argument('--olfactory', action='store_true',
                        help='Enable olfactory (attractive/repulsive odors)')
    parser.add_argument('--visual', action='store_true',
                        help='Enable camera vision: Q1 Lite eyes → T2 → LC4 → GF escape')
    parser.add_argument('--no-vision', action='store_true',
                        help='Disable visual rendering (faster, brain-only)')
    parser.add_argument('--consciousness', action='store_true',
                        help='Enable consciousness proxy measurement')
    parser.add_argument('--mlx', action='store_true',
                        help='Use MLX (Apple Silicon Metal) brain backend instead of PyTorch')
    parser.add_argument('--fast', action='store_true',
                        help='Speed mode: skip Hebbian plasticity, fewer brain substeps')
    parser.add_argument('--mg90s', action='store_true',
                        help='Model MG90S PWM servos: 50Hz commands, weaker '
                             'internal loop, stall-limited torque, no joint '
                             'readback (taste FK uses commanded angles)')
    parser.add_argument('--servo', action='store_true',
                        help='Mirror CPG joint targets onto real PWM servos '
                             '(PCA9685, 8 channels) on the Raspberry Pi '
                             '(loop becomes real-time paced)')
    parser.add_argument('--servo-joint', default=None, choices=JOINTS,
                        help='mirror only this joint (default: all 8 joints)')
    parser.add_argument('--servo-host', default='ubuntu@192.168.1.141',
                        help='SSH target running servo_stream.py')
    parser.add_argument('--servo-center', type=float, default=90.0,
                        help='servo deg at the joint standby pose')
    parser.add_argument('--servo-scale', type=float, default=57.29578,
                        help='servo deg per rad (negative if mounted reversed)')
    parser.add_argument('--servo-map', default=None,
                        help='JSON per-joint overrides, e.g. '
                             '\'{"FL_hip": {"center": 90, "scale": -57.3}}\'')
    parser.add_argument('--camera', choices=['virtual', 'real'],
                        default='virtual',
                        help='camera view: virtual = MuJoCo viewer (default), '
                             'real = Raspberry Pi USB camera over SSH MJPEG '
                             '(replaces the MuJoCo viewer window)')
    parser.add_argument('--camera-host', default=PI_CAM_HOST,
                        help='SSH target running camera_stream.py '
                             '(real camera mode)')
    parser.add_argument('--camera-port', type=int, default=PI_CAM_PORT,
                        help='MJPEG stream port on the Pi (real camera mode)')
    args = parser.parse_args()

    # NOTE on launchers (macOS): the real-camera display runs in a spawned
    # child process, so --camera real works under BOTH plain python and
    # mjpython. Under mjpython the MuJoCo viewer ALSO opens (sim state +
    # camera side by side); under plain python only the camera window
    # opens (launch_passive needs mjpython).

    project_root = Path(__file__).resolve().parent

    # ── State ──────────────────────────────────────────────────────────
    active_stimulus = [args.stimulus or 'p9']
    stim_changed = [True]
    auto_demo_enabled = [not args.no_auto and args.stimulus is None]

    demo_idx = [0]
    demo_time_remaining = [AUTO_DEMO_SEQUENCE[0][1]]

    # Keyboard mapping
    KEY_MAP = {
        ord('1'): 'sugar', ord('2'): 'p9', ord('3'): 'lc4',
        ord('4'): 'jo',    ord('5'): 'bitter', ord('6'): 'or56a',
        ord('0'): None,
    }
    GLFW_KEY_SPACE = 32

    def key_callback(keycode):
        if keycode == GLFW_KEY_SPACE:
            auto_demo_enabled[0] = not auto_demo_enabled[0]
            state = "ON" if auto_demo_enabled[0] else "OFF"
            print(f"\n[Auto-demo] {state}")
            if auto_demo_enabled[0]:
                demo_idx[0] = 0
                demo_time_remaining[0] = AUTO_DEMO_SEQUENCE[0][1]
            return
        if keycode in KEY_MAP:
            auto_demo_enabled[0] = False
            active_stimulus[0] = KEY_MAP[keycode]
            stim_changed[0] = True
            name = active_stimulus[0]
            if name and name in STIMULI:
                print(f"\n[Manual] {STIMULI[name]['description']}")
            else:
                print("\n[Manual] OFF — spontaneous activity")

    # ── Initialize Brain ───────────────────────────────────────────────
    brain = None
    if not args.no_brain:
        if args.mlx:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
            from brain_body_bridge_mlx import MlxBrainEngine
            print("Initializing brain (138,639 neurons on Metal via MLX)...")
            brain = MlxBrainEngine()
        else:
            print("Initializing brain (138,639 neurons on GPU)...")
            brain = BrainEngine(device='cuda')

    # ── Initialize Body (Q1 Lite) ─────────────────────────────────────────
    print(f"Initializing Q1 Lite body from {_Q1LITE_SCENE}...")
    sim = Q1LiteSim(str(_Q1LITE_SCENE), timestep=0.001)
    sim.reset()
    print(f"Q1 Lite: {sim.model.nu} actuators, standing at z={sim.position[2]:.3f}m")

    if args.mg90s:
        sim.set_servo_model(kp=2.0, kd=0.05, torque_limit=0.2, cmd_rate_hz=50.0)
        print("MG90S servo model: 50Hz latched commands, internal PD "
              "(kp=2.0, kd=0.05), stall-limited to 0.20 N·m — no joint readback")

    # ── Connect Real Camera (before visual init: the compound eyes may
    #    use it as input) ─────────────────────────────────────────────
    pi_camera = None
    if args.camera == 'real':
        print(f"Launching Raspberry Pi camera stream on {args.camera_host}...")
        try:
            pi_camera = PiCameraViewer(
                host=args.camera_host, port=args.camera_port,
                window='Q1 Lite — Real Camera (Pi)')
        except RuntimeError as e:
            print(f"[WARN] real camera failed ({e}); "
                  f"falling back to virtual eyes/viewer.")
            pi_camera = None

    # ── Initialize Visual System ─────────────────────────────────────────
    visual = None
    cached_visual = (None, None)
    last_vision_obs = None
    VISION_RATIO = 500  # process vision every 500 physics steps (500ms)
    if args.visual and brain is not None:
        if pi_camera is not None:
            print("Initializing visual system "
                  "(REAL Pi camera eyes → T2 → LC4 → GF)...")
            q1lite_vision = RealCameraVisualBridge(
                pi_camera, width=128, height=128)
        else:
            print("Initializing visual system (Q1 Lite eyes → T2 → LC4 → GF)...")
            q1lite_vision = Q1LiteVisualBridge(sim.model, sim.data, width=128, height=128)
        visual = VisualSystem(brain.flyid2i, brain.i2flyid)
        print(f"  T2 neurons: {visual._n_T2 if hasattr(visual, '_n_T2') else '?'} "
              f"LC4: {sum(len(v) for v in visual.get_lc4_indices(brain.flyid2i).values())} "
              f"LPLC2: {sum(len(v) for v in visual.get_lplc2_indices(brain.flyid2i).values())}")

    # ── Initialize Sensory Systems ─────────────────────────────────────
    somato = None
    vibration_sources = []
    if args.somatosensory and brain is not None:
        print("Initializing somatosensory system (JO vibration/sound — no contact force)...")
        somato = SomatosensorySystem(brain.flyid2i)
        vibration_sources = [
            VibrationSource(
                position=[320, 200, 5],
                frequency=200.0, amplitude=0.8, label='courtship'),
            VibrationSource(
                position=[-230, -160, 5],
                frequency=400.0, amplitude=0.6, label='alarm'),
        ]
        for vs in vibration_sources:
            print(f"  Vibration: '{vs.label}' at [{vs.position[0]:.1f},{vs.position[1]:.1f}]mm")

    taste_zones = []
    gusto = None
    if args.gustatory and brain is not None:
        print("Initializing gustatory system (sugar/bitter zones)...")
        # Positions are in mm — the gustatory/olfactory systems and the
        # Q1 Lite -> fly adapters all work in mm (sim.position * 1000).
        taste_zones = [
            TasteZone(center=[110, 260], radius=100,
                      taste='sugar', label='sugar_patch'),
            TasteZone(center=[320, -200], radius=80,
                      taste='bitter', label='bitter_patch'),
        ]
        # Q1 Lite has 4 real feet; LM/RM are geometric midpoints (phantom legs).
        # Tell gustatory so phantom legs can't fabricate independent taste
        # contact or inflate the leg count (dedup to the 4 real feet).
        gusto = GustatorySystem(
            brain.flyid2i, taste_zones,
            derived_legs={'LM': ('LF', 'LH'), 'RM': ('RF', 'RH')},
            ground_z_thresh=7.0)  # mm: Q1 Lite grounded foot ~3mm, swing lift ~11mm

        # Place the taste-zone floor patches to match these zones (match
        # the fly LoomingArena visuals, dynamically positioned)
        sim.place_taste_zones(taste_zones)

    olfact = None
    odor_sources = []
    if args.olfactory and brain is not None:
        print("Initializing olfactory system (Or42b + Or56a)...")
        odor_sources = [
            OdorSource(
                position=[380, 150, 5],
                odor_type='attractive', amplitude=0.9, spread=250, label='food'),
            OdorSource(
                position=[-260, -210, 5],
                odor_type='repulsive', amplitude=0.8, spread=200, label='geosmin'),
        ]
        olfact = OlfactorySystem(brain.flyid2i, antenna_spread=40.0)  # mm: Q1 Lite head width
        for src in odor_sources:
            print(f"  Odor: '{src.label}' ({src.odor_type}) at [{src.position[0]:.0f},{src.position[1]:.0f}]mm")

        # Place the odor-source orbs + halos to match these sources (match
        # the fly LoomingArena visuals, dynamically positioned)
        sim.place_odor_sources(odor_sources)

    # ── Initialize Consciousness ───────────────────────────────────────
    consciousness = None
    if args.consciousness and brain is not None:
        if ConsciousnessDetector is not None:
            consciousness = ConsciousnessDetector(brain)
        else:
            print("[WARN] consciousness.py not found, --consciousness ignored")

    # ── Initialize Bridge ──────────────────────────────────────────────
    decoder = DNRateDecoder(window_ms=50.0, dt_ms=0.1, max_rate=200.0)
    adaptor = Q1LiteAdaptor(decoder, dt=0.01)

    # ── Real-servo mirror (optional: 8 joints onto the Pi) ───────────
    servo = None
    if args.servo:
        joints_list = None if args.servo_joint is None else [args.servo_joint]
        all_mirrored = JOINTS if joints_list is None else joints_list
        centers = {j: args.servo_center for j in all_mirrored}
        scales = {j: args.servo_scale for j in all_mirrored}
        if args.servo_map:
            for j, ov in json.loads(args.servo_map).items():
                centers[j] = float(ov.get('center', args.servo_center))
                scales[j] = float(ov.get('scale', args.servo_scale))
        print(f"Connecting servo mirror: "
              f"{'all 8 joints' if joints_list is None else joints_list[0]} "
              f"-> {args.servo_host} ...")
        servo = ServoMirror(joints=joints_list, host=args.servo_host,
                            centers=centers, scales=scales)
        if args.servo_joint is None:
            print(f"Servo mirror active: all 8 joints -> PCA9685 ch0-7 "
                  f"@ {args.servo_host} (50Hz)")
        else:
            leg, part = args.servo_joint.split('_')
            stand = HIP0[leg] if part == 'hip' else 0.0
            print(f"Servo mirror active: {args.servo_joint} "
                  f"(stand {np.degrees(stand):.0f} deg -> "
                  f"servo {centers[args.servo_joint]:.0f} deg, 50Hz)")

    # Register populations for JO monitoring (sound only: no contact-force
    # sensing on the v1 hardware target, so the JO touch channel stays silent)
    if somato is not None and brain is not None:
        if len(somato.sound_idx_left) > 0:
            brain.register_population('JO_sound_L', somato.sound_idx_left)
            decoder.register_population('JO_sound_L')
        if len(somato.sound_idx_right) > 0:
            brain.register_population('JO_sound_R', somato.sound_idx_right)
            decoder.register_population('JO_sound_R')

    # Register LPLC2/LC4 populations for directional escape
    if args.visual and visual is not None:
        lplc2_idx = visual.get_lplc2_indices(brain.flyid2i)
        lc4_idx = visual.get_lc4_indices(brain.flyid2i)
        for name, indices in {**lplc2_idx, **lc4_idx}.items():
            brain.register_population(name, indices)
            decoder.register_population(name)

    # ── Set initial stimulus ───────────────────────────────────────────
    if brain is not None:
        brain.set_stimulus(active_stimulus[0])
        stim_changed[0] = False
        stim_desc = STIMULI.get(active_stimulus[0], {}).get(
            'description', active_stimulus[0] or 'none')
        print(f"Initial stimulus: {stim_desc}")

    # ── Timing Constants ───────────────────────────────────────────────
    PHYSICS_DT = 0.001          # 1 ms (1000 Hz)
    BRAIN_RATIO = 10            # 1 brain bundle per 10 physics steps = every 10ms
    BRAIN_SUBSTEPS = 3 if args.fast else 5  # LIF steps per bundle (more = stronger DN signal)
    MONITOR_INTERVAL = 50       # send brain data every 50 brain bundles (~0.5s)
    STATUS_INTERVAL = 1000      # status print every 1000 physics steps (1.0s)

    step = 0
    prev_mode = 'walking'
    fwd_drive, turn_drive = 0.0, 0.0
    stand_pose = QuadCPG(dt=PHYSICS_DT).stand_offsets.copy()

    # ── Launch Viewer ──────────────────────────────────────────────────
    # MuJoCo viewer shows the SIM STATE in both camera modes (under
    # mjpython); the real-camera window (pi_camera, spawned child) shows
    # the live Pi camera feed in --camera real mode.
    viewer = None
    if not args.no_viewer:
        print("Launching MuJoCo viewer...")
        try:
            viewer = sim.launch_viewer('Q1 Lite Brain-Body')
        except RuntimeError as e:
            # launch_passive needs mjpython on macOS; under plain python
            # fall back gracefully (camera-only or headless)
            print(f"[WARN] MuJoCo viewer unavailable ({e}); "
                  f"continuing without it.")
            viewer = None

    # ── Launch Brain Monitor ───────────────────────────────────────────
    monitor = None
    if args.monitor:
        print("Launching brain monitor...")
        # macOS requires 'spawn' for pygame child processes
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # already set
        monitor = BrainMonitorProcess()
        monitor.start()

    # ── Main Loop ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  EMBODIED BRAIN → Q1 LITE SPIDER QUADRUPED")
    print(f"  Brain: 138,639 LIF neurons on GPU")
    print(f"  Body:  Q1 Lite spider quadruped, {sim.model.nu} actuators, physics @ {PHYSICS_DT*1000:.0f}ms")
    print(f"  Neural: 1 brain step × {BRAIN_RATIO} phys steps = {BRAIN_RATIO*PHYSICS_DT*1000:.0f}ms interval")
    if args.mg90s:
        print("  Servo:  MG90S PWM model (50Hz commands, no readback)")
    if args.servo:
        if args.servo_joint is None:
            print(f"  Mirror: 8 joints -> PCA9685 @ {args.servo_host} "
                  f"(real-time paced)")
        else:
            print(f"  Mirror: {args.servo_joint} -> PCA9685 @ {args.servo_host} "
                  f"(real-time paced)")
    if pi_camera is not None:
        print(f"  Camera: REAL — Pi USB camera @ {args.camera_host} "
              f"(compound eyes + live view)")
        if args.visual and brain is not None:
            print("  Eyes:   real camera frame → T2 → LC4 → GF")
    elif viewer is not None:
        print("  Camera: virtual (MuJoCo eyes + viewer)")
    if viewer is not None and pi_camera is not None:
        print("  MuJoCo: viewer ON (sim state) alongside camera window")
    if auto_demo_enabled[0]:
        print("  MODE: Auto-demo (SPACE to toggle)")
    else:
        print("  MODE: Manual (keys: 1=sugar 2=P9 3=LC4 4=JO 5=bitter 6=Or56a 0=off)")
    if pi_camera is not None and viewer is not None:
        print("  Close either window to exit (q/ESC works in camera window)")
    elif pi_camera is not None:
        print("  Press q / ESC in camera window to exit")
    elif viewer is not None:
        print("  Close viewer to exit")
    else:
        print("  Headless — Ctrl-C to exit")
    print("=" * 70)
    print()

    braindata_sender = None

    # Wall-clock reference for real-time pacing (used with --servo)
    t0_wall = time.time()

    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if pi_camera is not None and not pi_camera.is_running():
                print(f"\n[CAMERA] Pi camera stream ended unexpectedly — "
                      f"{pi_camera.explain_exit()}")
                print("[CAMERA] (Pi reboot / WiFi drop / camera read "
                      "failure are the usual causes; check the Pi and "
                      "rerun.)")
                break
            if args.duration > 0 and step * PHYSICS_DT >= args.duration:
                break

            # ── Auto-demo ───────────────────────────────────────────
            if auto_demo_enabled[0] and brain is not None:
                demo_time_remaining[0] -= PHYSICS_DT
                if demo_time_remaining[0] <= 0:
                    demo_idx[0] = (demo_idx[0] + 1) % len(AUTO_DEMO_SEQUENCE)
                    stim_name, duration, desc = AUTO_DEMO_SEQUENCE[demo_idx[0]]
                    demo_time_remaining[0] = duration
                    active_stimulus[0] = stim_name
                    stim_changed[0] = True
                    print(f"\n  >>> [{desc}] ({stim_name or 'none'}, {duration:.1f}s)")

            # ── Update stimulus ─────────────────────────────────────
            if stim_changed[0] and brain is not None:
                brain.set_stimulus(active_stimulus[0])
                stim_changed[0] = False
                # Re-apply cached visual rates (set_stimulus zeroes all rates)
                if cached_visual[0] is not None:
                    brain.set_visual_rates(*cached_visual)

            # ── Visual processing (every VISION_RATIO steps) ──────────
            do_vision = (args.visual and visual is not None and step % VISION_RATIO == 0
                         and not args.no_vision)
            if do_vision:
                # Move looming ball toward robot along +Y (forward).
                # Robot-proportioned ball (r=50mm ~= 1/3 of the 157mm
                # stand diagonal, center at eye height z=50mm): approaches
                # at 0.04 m/s (~2x walking speed), 0.6m -> 0.12m sawtooth
                # (min distance keeps the eyes outside the ball surface).
                ball_dist = 0.6 - (step * PHYSICS_DT * 0.04) % 0.48
                ball_pos = np.array([sim.position[0],
                                     sim.position[1] + ball_dist, 0.05])
                sim.set_looming_ball(ball_pos)

                vision_obs = q1lite_vision.process()

                vis_idx, vis_rates = visual.process_visual_layers(vision_obs)
                if vis_idx is not None:
                    cached_visual = (vis_idx, vis_rates)
                    brain.set_visual_rates(vis_idx, vis_rates)
                # Cache vision_obs for monitor retina display
                last_vision_obs = vision_obs
                # Per-eye T2 fallback for directional threat bias
                if cached_visual[1] is not None and hasattr(visual, '_T2_eye'):
                    vis_eye = visual._T2_eye
                    vis_r = cached_visual[1]
                    mask_L = vis_eye == 0
                    mask_R = vis_eye == 1
                    t2_left = float(np.mean(vis_r[mask_L])) if mask_L.any() else 0.0
                    t2_right = float(np.mean(vis_r[mask_R])) if mask_R.any() else 0.0
                    adaptor.bridge.visual_threat_bias = (
                        (t2_right - t2_left) / (t2_left + t2_right + 1e-6))

            # ── Sensory processing (every brain interval) ───────────
            if step % BRAIN_RATIO == 0:
                brain_bundle = True

                # -- Somatosensory (vibration/sound only — no contact force) --
                if somato is not None:
                    # Vibration: Q1 Lite position (m → mm), heading
                    fly_pos_mm = sim.position * 1000.0
                    fly_heading = sim.heading_angle
                    somato.process_vibration(fly_pos_mm, fly_heading, vibration_sources)

                    jo_idx, jo_rates = somato.get_rates()
                    brain.set_sensory_rates(jo_idx, jo_rates)

                # -- Gustatory --
                if gusto is not None:
                    fly_pos_mm = sim.position * 1000.0
                    # MG90S mode: FK from latched commands (no joint readback)
                    feet = (sim.commanded_foot_positions if args.mg90s
                            else sim.foot_positions)
                    end_effectors = q1lite_feet_to_fly_end_effectors(
                        feet, sim.position)
                    gusto.process(end_effectors)

                    grn_idx, grn_rates = gusto.get_rates()
                    brain.set_sensory_rates(grn_idx, grn_rates)

                # -- Olfactory --
                if olfact is not None:
                    fly_pos_mm = sim.position * 1000.0
                    fly_heading = sim.heading_angle
                    olfact.process(fly_pos_mm, fly_heading, odor_sources)

                    or_idx, or_rates = olfact.get_rates()
                    brain.set_sensory_rates(or_idx, or_rates)

                # -- Brain step: BRAIN_SUBSTEPS × 0.1ms LIF per bundle --
                if brain is not None:
                    dn_accum = {name: 0.0 for name in decoder.dn_names}
                    for _ in range(BRAIN_SUBSTEPS):
                        brain.step()
                        spikes = brain.get_dn_spikes()
                        for name in dn_accum:
                            dn_accum[name] = max(dn_accum[name], spikes.get(name, 0.0))
                    pop_spikes = (brain.get_population_spikes()
                                  if brain.populations else None)
                    decoder.update(dn_accum, pop_spikes)

                    if consciousness is not None:
                        consciousness.update(step, adaptor.mode)

                # -- Sensory → Bridge state --
                if somato is not None:
                    adaptor.bridge.sound_orientation_bias = somato.orientation_bias
                if gusto is not None:
                    adaptor.bridge.bitter_active = gusto.bitter_active
                if olfact is not None:
                    adaptor.bridge.olfactory_attraction_bias = olfact.attraction_bias
                    adaptor.bridge.olfactory_repulsive = olfact.is_repulsive_escape
                    adaptor.bridge.olfactory_repulsion_bias = olfact.repulsion_bias

                # -- Compute joint targets ───────────────────────────
                fwd_drive, turn_drive = adaptor.compute_drive(
                    dt=BRAIN_RATIO * PHYSICS_DT)

                # -- Mode logging ────────────────────────────────────
                if adaptor.mode != prev_mode:
                    print(f"  >> Behavior: {prev_mode} -> {adaptor.mode}")
                    prev_mode = adaptor.mode

            else:
                brain_bundle = False

            # ── Body step: CPG at physics rate (every step) ───────
            if brain is not None:
                targets = adaptor.cpg.step(fwd_drive, turn_drive)
            else:
                targets = stand_pose
            sim.step(targets)
            if servo is not None:
                servo.send(targets)   # rate-limited to 50Hz internally

            # ── Status print ───────────────────────────────────────────
            if step % STATUS_INTERVAL == 0:
                pos = sim.position
                heading = np.degrees(sim.heading_angle)
                print(f"  [t={step*PHYSICS_DT:.1f}s] "
                      f"pos=({pos[0]:.2f},{pos[1]:.2f})m "
                      f"heading={heading:.0f} deg "
                      f"{adaptor.get_status_str() if brain is not None else ''}")

            # ── Brain Monitor data ─────────────────────────────────────
            if monitor is not None and brain is not None and step % (BRAIN_RATIO * MONITOR_INTERVAL) == 0:
                d = decoder
                mon_data = {
                    't_sim': step * PHYSICS_DT,
                    'mode': adaptor.mode,
                    'drive': [adaptor.drive[0], adaptor.drive[1]],
                    'stimulus': active_stimulus[0] or 'none',
                    'dn_forward': d.get_group_rate('forward'),
                    'dn_escape': d.get_group_rate('escape'),
                    'dn_groom': d.get_group_rate('groom'),
                    'dn_backward': d.get_group_rate('backward'),
                    'dn_feed': d.get_group_rate('feed'),
                    'dn_turn_L': d.get_group_rate('turn_L'),
                    'dn_turn_R': d.get_group_rate('turn_R'),
                }
                if somato is not None:
                    mon_data['jo_sound'] = somato.sound_level
                if gusto is not None:
                    mon_data['sugar_level'] = gusto.sugar_level
                    mon_data['bitter_level'] = gusto.bitter_level
                if olfact is not None:
                    mon_data['or_attractive'] = olfact.attractive_level
                    mon_data['or_repulsive'] = olfact.repulsive_level
                # Visual data
                if args.visual and visual is not None:
                    mon_data['lplc2_left'] = d.get_pop_rate('LPLC2_left')
                    mon_data['lplc2_right'] = d.get_pop_rate('LPLC2_right')
                    mon_data['lc4_left'] = d.get_pop_rate('LC4_left')
                    mon_data['lc4_right'] = d.get_pop_rate('LC4_right')
                    mon_data['threat_asym'] = adaptor.bridge.threat_asym
                    if cached_visual[1] is not None and hasattr(visual, '_T2_eye'):
                        vis_eye = visual._T2_eye
                        vis_r = cached_visual[1]
                        mask_L = vis_eye == 0
                        mask_R = vis_eye == 1
                        mon_data['t2_left'] = float(
                            np.mean(vis_r[mask_L]) / 120.0) if mask_L.any() else 0.0
                        mon_data['t2_right'] = float(
                            np.mean(vis_r[mask_R]) / 120.0) if mask_R.any() else 0.0
                    ball_pos = sim.get_looming_ball_pos()
                    if ball_pos is not None:
                        mon_data['ball_x'] = float(ball_pos[1])  # forward axis = +Y
                    # Raw eye frames for monitor compound-eye panel
                    rgb_l, rgb_r = q1lite_vision.get_eye_images()
                    if rgb_l is not None:
                        mon_data['eye_left'] = rgb_l
                        mon_data['eye_right'] = rgb_r
                    # Retina brightness (for monitor retina panel)
                    if last_vision_obs is not None:
                        mon_data['bright_left'] = float(np.mean(last_vision_obs[0]))
                        mon_data['bright_right'] = float(np.mean(last_vision_obs[1]))
                        mon_data['dark_omm_left'] = int(np.sum(
                            np.mean(last_vision_obs[0], axis=1) < 0.25))
                        mon_data['dark_omm_right'] = int(np.sum(
                            np.mean(last_vision_obs[1], axis=1) < 0.25))
                # Consciousness data
                if consciousness is not None:
                    mon_data.update(consciousness.get_monitor_data())
                monitor.send(mon_data)

            step += 1

            # ── Viewer / camera sync ──────────────────────────────────
            if viewer is not None:
                viewer.sync()
            if pi_camera is not None:
                if not pi_camera.show(on_key=key_callback):
                    print(f"\n[CAMERA] display closed — "
                          f"{pi_camera.close_reason}")
                    break

            # ── Real-time pacing (only when mirroring a real servo) ─
            if servo is not None:
                now = time.time()
                target_t = t0_wall + step * PHYSICS_DT
                if now < target_t:
                    time.sleep(target_t - now)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if pi_camera is not None:
            pi_camera.close()
        if viewer is not None:
            viewer.close()
        if servo is not None:
            servo.close()
        if brain is not None:
            brain.save_plastic_weights()
        if monitor is not None:
            monitor.stop()
        if consciousness is not None and hasattr(consciousness, 'save'):
            consciousness.save()
        print("Shutdown complete.")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    main()
