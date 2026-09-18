#!/usr/bin/env python3
"""
Embodied Drosophila → Unitree Go2: Brain-body closed loop.

Replaces the NeuroMechFly body with a Unitree Go2 quadruped.
Same 138,639-neuron brain, same sensory modules, different body.

Usage:
    python fly_embodied_unitree.py                        # Auto-demo
    python fly_embodied_unitree.py --stimulus p9          # Manual stimulus
    python fly_embodied_unitree.py --no-auto              # Keyboard only
    python fly_embodied_unitree.py --olfactory --gustatory --somatosensory

Keys (in MuJoCo viewer window):
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
import numpy as np
import mujoco
import multiprocessing as mp
from pathlib import Path

from brain_body_bridge import (
    BrainEngine, DNRateDecoder, STIMULI,
)
from unitree_bridge.go2_sim import Go2Sim
from unitree_bridge.go2_adaptor import Go2Adaptor, QuadCPG
from unitree_bridge.go2_vision import Go2VisualBridge

from visual_system import VisualSystem
from somatosensory import SomatosensorySystem, VibrationSource

from somatosensory import SomatosensorySystem, VibrationSource
from gustatory import GustatorySystem, TasteZone
from olfactory import OlfactorySystem, OdorSource
from brain_monitor import BrainMonitorProcess

try:
    from consciousness import ConsciousnessDetector
except ImportError:
    ConsciousnessDetector = None

# Go2 model path (Menagerie MJX: built-in PD servos, better friction/damping)
_GO2_SCENE = Path('./unitree_go2/scene.xml').resolve()

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
# Sensory Adaptors — convert Go2 sensor data to fly-compatible formats
# ============================================================================

def go2_contact_to_fly(go2_contact_forces):
    """
    Convert Go2 (4,) contact forces to fly-compatible (36, 3) array.

    Go2 has 4 feet, each with 1 scalar force.
    Fly interface expects (6 legs × 6 segments, 3 axes).
    We distribute the Go2 forces to create bilateral activation.
    """
    # Normalize Go2 forces
    fl_f  = max(go2_contact_forces[0], 0.0)  # FL foot
    fr_f  = max(go2_contact_forces[1], 0.0)  # FR foot
    rl_f  = max(go2_contact_forces[2], 0.0)  # RL foot
    rr_f  = max(go2_contact_forces[3], 0.0)  # RR foot

    # Per-foot → per-fly-leg assignment (preserves fore/aft load):
    #   LF ← FL,  LH ← RL  (front vs hind stay distinct, no averaging dilution)
    #   LM ← mean(FL, RL)  (phantom middle leg)
    #   RF ← FR,  RH ← RR,  RM ← mean(FR, RR)
    leg_force = np.array([
        fl_f, 0.5 * (fl_f + rl_f), rl_f,   # LF, LM, LH
        fr_f, 0.5 * (fr_f + rr_f), rr_f,   # RF, RM, RH
    ], dtype=np.float64)

    # Create (36, 3) array matching fly layout:
    # 6 legs (LF,LM,LH,RF,RM,RH) × 6 segments × 3 axes
    out = np.zeros((36, 3), dtype=np.float64)
    for leg in range(6):
        out[leg * 6:(leg + 1) * 6, 2] = leg_force[leg]  # z-axis force (vertical)

    return out


def go2_feet_to_fly_end_effectors(go2_foot_positions, sim_position):
    """
    Convert Go2 (4, 3) foot positions (m) to fly-compatible (6, 3) array (mm).

    Go2 feet: FL, FR, RL, RR (in m)
    Fly expects: 6 end effectors in mm
    We map: LF→FL, LM→(FL+RL)/2, LH→RL, RF→FR, RM→(FR+RR)/2, RH→RR
    """
    # Convert m to mm
    feet_mm = go2_foot_positions * 1000.0
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
    parser = argparse.ArgumentParser(description='Embodied Drosophila → Go2')
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
                        help='Enable touch/sound via JO neurons')
    parser.add_argument('--gustatory', action='store_true',
                        help='Enable taste zones (sugar/bitter)')
    parser.add_argument('--olfactory', action='store_true',
                        help='Enable olfactory (attractive/repulsive odors)')
    parser.add_argument('--visual', action='store_true',
                        help='Enable camera vision: Go2 eyes → T2 → LC4 → GF escape')
    parser.add_argument('--consciousness', action='store_true',
                        help='Enable consciousness proxy measurement')
    parser.add_argument('--mlx', action='store_true',
                        help='Use MLX (Apple Silicon Metal) brain backend instead of PyTorch')
    parser.add_argument('--no-vision', action='store_true',
                        help='Disable visual rendering (faster, brain-only)')
    parser.add_argument('--fast', action='store_true',
                        help='Speed mode: skip Hebbian plasticity, fewer brain substeps')
    args = parser.parse_args()

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

    # ── Initialize Body (Go2) ──────────────────────────────────────────
    print(f"Initializing Go2 body from {_GO2_SCENE}...")
    sim = Go2Sim(str(_GO2_SCENE), timestep=0.001)
    sim.reset()
    print(f"Go2: {sim.model.nu} actuators, standing at z={sim.position[2]:.3f}m")

    # ── Initialize Visual System ─────────────────────────────────────────
    visual = None
    cached_visual = (None, None)
    last_vision_obs = None
    VISION_RATIO = 500  # process vision every 500 physics steps (500ms)
    if args.visual and brain is not None:
        print("Initializing visual system (Go2 cameras → T2 → LC4 → GF)...")
        go2_vision = Go2VisualBridge(sim.model, sim.data, width=128, height=128)
        visual = VisualSystem(brain.flyid2i, brain.i2flyid)
        print(f"  T2 neurons: {visual._n_T2 if hasattr(visual, '_n_T2') else '?'} "
              f"LC4: {sum(len(v) for v in visual.get_lc4_indices(brain.flyid2i).values())} "
              f"LPLC2: {sum(len(v) for v in visual.get_lplc2_indices(brain.flyid2i).values())}")

    # ── Initialize Sensory Systems ─────────────────────────────────────
    somato = None
    vibration_sources = []
    if args.somatosensory and brain is not None:
        print("Initializing somatosensory system (JO touch + sound)...")
        somato = SomatosensorySystem(brain.flyid2i)
        vibration_sources = [
            VibrationSource(
                position=[3.0, 2.0, 0.05],
                frequency=200.0, amplitude=0.8, label='courtship'),
            VibrationSource(
                position=[-2.0, -1.5, 0.05],
                frequency=400.0, amplitude=0.6, label='alarm'),
        ]
        for vs in vibration_sources:
            print(f"  Vibration: '{vs.label}' at [{vs.position[0]:.1f},{vs.position[1]:.1f}]m")

    taste_zones = []
    gusto = None
    if args.gustatory and brain is not None:
        print("Initializing gustatory system (sugar/bitter zones)...")
        taste_zones = [
            TasteZone(center=[1.5, 0.5], radius=0.8,
                      taste='sugar', label='sugar_patch'),
            TasteZone(center=[2.0, -1.0], radius=0.6,
                      taste='bitter', label='bitter_patch'),
        ]
        # Go2 has 4 real feet; LM/RM are geometric midpoints (phantom legs).
        # Tell gustatory so phantom legs can't fabricate independent taste
        # contact or inflate the leg count (dedup to the 4 real feet).
        gusto = GustatorySystem(
            brain.flyid2i, taste_zones,
            derived_legs={'LM': ('LF', 'LH'), 'RM': ('RF', 'RH')})

    olfact = None
    odor_sources = []
    if args.olfactory and brain is not None:
        print("Initializing olfactory system (Or42b + Or56a)...")
        odor_sources = [
            OdorSource(
                position=[2.5, 1.0, 0.05],
                odor_type='attractive', amplitude=0.9, spread=2.5, label='food'),
            OdorSource(
                position=[-1.5, -1.2, 0.05],
                odor_type='repulsive', amplitude=0.8, spread=2.0, label='geosmin'),
        ]
        olfact = OlfactorySystem(brain.flyid2i)
        for src in odor_sources:
            print(f"  Odor: '{src.label}' ({src.odor_type}) at [{src.position[0]:.1f},{src.position[1]:.1f}]m")

    # ── Initialize Consciousness ───────────────────────────────────────
    consciousness = None
    if args.consciousness and brain is not None:
        if ConsciousnessDetector is not None:
            consciousness = ConsciousnessDetector(brain)
        else:
            print("[WARN] consciousness.py not found, --consciousness ignored")

    # ── Initialize Bridge ──────────────────────────────────────────────
    decoder = DNRateDecoder(window_ms=50.0, dt_ms=0.1, max_rate=200.0)
    adaptor = Go2Adaptor(decoder, dt=0.01)

    # Register populations for JO monitoring
    if somato is not None and brain is not None:
        if len(somato.touch_idx_left) > 0:
            brain.register_population('JO_touch_L', somato.touch_idx_left)
            decoder.register_population('JO_touch_L')
        if len(somato.touch_idx_right) > 0:
            brain.register_population('JO_touch_R', somato.touch_idx_right)
            decoder.register_population('JO_touch_R')
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
    viewer = None
    if not args.no_viewer:
        print("Launching MuJoCo viewer...")
        viewer = sim.launch_viewer('Go2 Brain-Body')

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
    print("  EMBODIED BRAIN → UNITREE GO2")
    print(f"  Brain: 138,639 LIF neurons on GPU")
    print(f"  Body:  Go2 quadruped, 12 actuators, physics @ {PHYSICS_DT*1000:.0f}ms")
    print(f"  Neural: 1 brain step × {BRAIN_RATIO} phys steps = {BRAIN_RATIO*PHYSICS_DT*1000:.0f}ms interval")
    if auto_demo_enabled[0]:
        print("  MODE: Auto-demo (SPACE to toggle)")
    else:
        print("  MODE: Manual (keys: 1=sugar 2=P9 3=LC4 4=JO 5=bitter 6=Or56a 0=off)")
    print("  Close viewer to exit")
    print("=" * 70)
    print()

    braindata_sender = None

    try:
        while True:
            if viewer is not None and not viewer.is_running():
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
                # Re-apply cached visual rates (set_stimulus zeros all rates)
                if cached_visual[0] is not None:
                    brain.set_visual_rates(*cached_visual)

            # ── Visual processing (every VISION_RATIO steps) ──────────
            do_vision = (args.visual and visual is not None and step % VISION_RATIO == 0
                         and not args.no_vision)
            if do_vision:
                # Move looming ball toward robot (cycles between far and near)
                ball_dist = 4.0 - (step * PHYSICS_DT * 0.5) % 3.8  # 0.5m/s approach, min 0.2m
                ball_pos = sim.position + np.array([ball_dist, 0.0, 0.25])
                sim.set_looming_ball(ball_pos)

                vision_obs = go2_vision.process()

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

                # -- Somatosensory --
                if somato is not None:
                    # Contact forces: Go2 (4,) → fly (36, 3)
                    contact_sim = sim.contact_forces
                    contact_fly = go2_contact_to_fly(contact_sim)
                    somato.process_contact(contact_fly)

                    # Vibration: Go2 position (m → mm), heading
                    fly_pos_mm = sim.position * 1000.0
                    fly_heading = sim.heading_angle
                    somato.process_vibration(fly_pos_mm, fly_heading, vibration_sources)

                    jo_idx, jo_rates = somato.get_rates()
                    brain.set_sensory_rates(jo_idx, jo_rates)

                # -- Gustatory --
                if gusto is not None:
                    fly_pos_mm = sim.position * 1000.0
                    end_effectors = go2_feet_to_fly_end_effectors(
                        sim.foot_positions, sim.position)
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
                    adaptor.bridge.tactile_force = somato.max_contact_force
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
                sim.step(adaptor.cpg.step(fwd_drive, turn_drive))
            else:
                sim.step(stand_pose)

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
                    mon_data['jo_contact'] = somato.touch_level
                    mon_data['jo_sound'] = somato.sound_level
                    mon_data['contact_force'] = somato.max_contact_force
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
                        mon_data['ball_x'] = float(ball_pos[0])
                    # Raw eye frames for monitor compound-eye panel
                    rgb_l, rgb_r = go2_vision.get_eye_images()
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

            # ── Viewer sync ────────────────────────────────────────────
            if viewer is not None:
                viewer.sync()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if viewer is not None:
            viewer.close()
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