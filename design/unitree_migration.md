# Fly-Brain → Unitree Go2 Migration Design

## Overview

Replace the NeuroMechFly v2 body (flygym + MuJoCo, 42-DOF fly) with a Unitree Go2 quadruped robot (12-DOF) while keeping the **entire 138,639-neuron brain simulation** unchanged.

| | Current (Fly) | Target (Go2) |
|---|---|---|
| Body model | flygym.Fly → MJCF | unitree_mujoco go2/scene.xml |
| DOF | 42 (6 legs × 7) | 12 (4 legs × 3) |
| Control | `HybridTurningController.step(drive)` | Direct `mj_data.ctrl` via PD |
| Physics dt | 0.1 ms | 1.0 ms |
| Brain ratio | 100 (10 Hz neural) | 10 (10 brain steps / physics step) |
| Simulation class | flygym.HybridTurningController | Custom `Go2Sim` wrapper |
| Viewer | mujoco.viewer.launch_passive | Same (same mujoco pkg) |

---

## Architecture: What Changes vs. What Stays

```
                         UNCHANGED                          │            MODIFIED
┌──────────────────────────────────────────────────────────┐│┌──────────────────────────────────┐
│ Neural (GPU, PyTorch)                                    │││ New: Go2Sim                     │
│ BrainEngine.step() @ 0.1ms                             │││ mujoco.MjModel.from_xml_path(  │
│ → DN spikes → DNRateDecoder → firing rates               │││   "unitree_robots/go2/scene")  │
│                                                          │││ mujoco.mj_step(m, d)           │
│ STIMULI / DN_NEURONS / MODEL_PARAMS — no change          │││ d.ctrl[:] = joint_targets      │
│ visual_system / somatosensory / olfactory / gustatory    │││                                 │
│ Hebbian plasticity / consciousness                       │││ READ: d.qpos, d.sensordata      │
└──────────────────────────────┬───────────────────────────┘│└──────────────┬───────────────────┘
                               │                            │               │
                               ▼                            │               ▼
┌──────────────────────────────────────────────────────────┐│┌──────────────────────────────────┐
│ BrainBodyBridge.compute_drive() — KEPT but output changes │││ New: Go2Adaptor                 │
│                                                          │││ [left_drive, right_drive]       │
│ → [left_drive, right_drive] ∈ [-0.5, 1.5]²  (kept)      │││   → 4-leg CPG                  │
│                                                          │││   → 12 joint targets (PD)      │
│ Escape / Groom / Feed modes → REMOVED                    │││                                 │
└──────────────────────────────────────────────────────────┘│└──────────────────────────────────┘
```

---

## File Plan

| Action | File | Purpose |
|--------|------|---------|
| **NEW** | `unitree_bridge/go2_sim.py` | Go2Sim: load model, step, sensors, viewer |
| **NEW** | `unitree_bridge/go2_adaptor.py` | Go2Adaptor: DN rates → CPG → 12 joint targets |
| **NEW** | `fly_embodied_unitree.py` | Main loop (replaces fly_embodied.py) |
| **NEW** | `design/unitree_migration.md` | This document |
| **KEEP** | `brain_body_bridge.py` | BrainEngine, DNRateDecoder, DN_NEURONS, STIMULI |
| **KEEP** | `code/run_pytorch.py` | TorchModel, get_weights, get_hash_tables |
| **KEEP** | `visual_system.py` | Photoreceptor mapping |
| **KEEP** | `somatosensory.py` | JO touch/sound (sensor shapes adapt) |
| **KEEP** | `gustatory.py` | GRN taste zones (end_effectors: 6→4) |
| **KEEP** | `olfactory.py` | ORN odor (position-only, unchanged) |
| **KEEP** | `consciousness.py` | Spike-only, unchanged |
| **REMOVE** | `flight.py`, `vocalization.py` | Fly-specific |
| **UNUSED** | `fly_embodied.py`, `two_flies.py`, `fly_alive.py`, `fly_walk.py`, `fly_behaviors.py` | Replaced |

---

## Phase 1: Install unitree_mujoco & Verify Model Loading

Install unitree_mujoco (clone repo for MJCF files, no DDS bridge needed):

```bash
cd /Users/dan/robot/fly-brain
git clone https://github.com/unitreerobotics/unitree_mujoco.git vendor/unitree_mujoco
```

Verify loading:

```python
import mujoco
model = mujoco.MjModel.from_xml_path("vendor/unitree_mujoco/unitree_robots/go2/scene.xml")
print(f"Go2: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
# Expected: nq=19 (7 free + 12 joints), nv=18, nu=12
```

---

## Phase 2: Go2Sim — Body Wrapper

A lightweight wrapper around raw MuJoCo. Responsibilities:

1. **Load** MJCF model from `go2/scene.xml`
2. **Step** physics: `mujoco.mj_step(m, d)`
3. **Control**: write `d.ctrl[:]` (12 PD target positions)
4. **Read sensors**: joint positions, IMU, contact forces
5. **Viewer**: `mujoco.viewer.launch_passive`

```python
class Go2Sim:
    def __init__(self, model_path, timestep=0.001):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def step(self, joint_targets):
        """joint_targets: np.ndarray of shape (12,) — PD target positions"""
        self.data.ctrl[:] = joint_targets
        mujoco.mj_step(self.model, self.data)

    @property
    def position(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()        # world x,y,z

    @property
    def orientation_quat(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()         # w,x,y,z

    @property
    def forward_vector(self) -> np.ndarray:
        """Body X-axis in world frame (heading direction)."""
        quat = self.orientation_quat
        # Rotate (1,0,0) by quaternion
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        fx = 1 - 2*(y*y + z*z)
        fy = 2*(x*y + w*z)
        return np.array([fx, fy, 0.0])

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[7:19].copy()        # 12 joint positions

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[6:18].copy()         # 12 joint velocities

    @property
    def foot_positions(self) -> np.ndarray:
        """World-frame positions of 4 feet from forward kinematics."""
        # Go2 foot body names (from scene.xml): FL_foot, FR_foot, RL_foot, RR_foot
        ...

    @property
    def contact_forces(self) -> np.ndarray:
        """4 foot contact forces from sensor data."""
        # Depends on Go2 scene.xml sensor naming
        ...
```

---

## Phase 3: Go2Adaptor — DN Rates → Joint Targets

Replaces the role of `HybridTurningController` + `BrainBodyBridge` (mode logic).

**Design**: Keep `[left_drive, right_drive]` as the intermediate representation, then adapt to 4 legs via a simple CPG.

```python
class Go2Adaptor:
    """
    Converts DN firing rates to Go2 joint targets.

    Pipeline:
      DNRateDecoder.rates → BrainBodyBridge.compute_drive() → [L, R]
      → QuadCPG → 12 joint targets
    """
    def __init__(self, dt=0.01):
        self.bridge = BrainBodyBridge(decoder)
        self.cpg = QuadCPG(dt=dt)
        # Standing pose (neutral joint positions)
        self.stand_pose = np.array([...])  # 12 values from Go2 standing config

    def compute_action(self, dn_rates, pop_rates=None):
        """Return 12-element joint target array."""
        drive = self.bridge.compute_drive()
        # drive[0] = forward, drive[1] = turn
        return self.cpg.step(drive[0], drive[1])
```

### QuadCPG Design

Simplified 4-leg CPG from the existing `WalkingCPG` in `two_flies.py`, adapted for Go2:

```
4 legs: FR (front-right), FL, RR, RL
Each leg: 3 DOF (hip_aa, hip_fe, knee_fe) — Go2 naming

Gait: Trot (diagonal pairs)
  Phase: FR + RL in sync, FL + RR in sync, 180° offset

Drive mapping:
  forward → amplitude scaling (0.0=stand, 1.0=full stride)
  turn    → left-right amplitude asymmetry
  negative forward → backward walking
```

Joint naming (Go2):
```
FR_hip_joint   (abduction/adduction)
FR_thigh_joint (hip flexion/extension)
FR_calf_joint  (knee)

FL_hip_joint, FL_thigh_joint, FL_calf_joint
RR_hip_joint, RR_thigh_joint, RR_calf_joint
RL_hip_joint, RL_thigh_joint, RL_calf_joint
```

---

## Phase 4: Main Loop

```python
# fly_embodied_unitree.py

def main():
    # --- Brain (unchanged) ---
    brain = BrainEngine(device='cuda')
    decoder = DNRateDecoder(window_ms=50.0, dt_ms=0.1)

    # --- Body (new) ---
    sim = Go2Sim(model_path="vendor/unitree_mujoco/unitree_robots/go2/scene.xml")
    adaptor = Go2Adaptor(decoder, dt=0.01)

    # --- Sensory modules (adapted) ---
    somato = SomatosensorySystem(brain.flyid2i)
    gusto = GustatorySystem(brain.flyid2i, taste_zones)
    olfact = OlfactorySystem(brain.flyid2i)

    # --- Viewer ---
    viewer = mujoco.viewer.launch_passive(model_ptr, data_ptr, ...)

    # --- Loop ---
    BRAIN_RATIO = 10  # 1 brain bundle per physics step (1ms → 10 × 0.1ms)

    while viewer.is_running():
        # Sensory → Brain
        fly_pos = sim.position
        fly_orient = sim.forward_vector  # heading
        contact_forces = sim.contact_forces
        foot_positions = sim.foot_positions

        somato.process_contact(contact_forces)
        somato.process_vibration(fly_pos, heading, sources)
        gusto.process(foot_positions)
        olfact.process(fly_pos, heading, sources)

        brain.set_sensory_rates(somato_indices, somato_rates)
        brain.set_sensory_rates(grn_indices, grn_rates)
        brain.set_sensory_rates(orn_indices, orn_rates)

        # Brain step (10 substeps)
        for _ in range(BRAIN_RATIO):
            brain.step()
        dn_spikes = brain.get_dn_spikes()
        decoder.update(dn_spikes)

        # Brain → Body
        joint_targets = adaptor.compute_action(decoder.rates)
        sim.step(joint_targets)

        viewer.sync()
```

---

## Phase 5: Sensory Remapping

### Somatosensory

`contact_forces` changes from (36, 3) to (4, 3) or a flat array from Go2 foot sensors.

`somatosensory.py` already reads a flat `contact_forces` array. Need to identify Go2 sensor indices.

### Gustatory

`GustatorySystem.process(end_effectors)` takes (6, 3) → change to (4, 3). The `TasteZone` logic is unchanged.

### Olfactory

No change needed — only uses `fly_pos` and heading.

### Visual

Defer to Phase 6+. Options:
- Use arena camera
- Mount virtual camera on Go2 head
- Skip for initial integration

---

## Phase 6: Integration Test

```bash
/Users/dan/miniconda3/envs/fly/bin/python fly_embodied_unitree.py \
    --olfactory --gustatory --somatosensory --monitor
```

Success criteria:
- Go2 robot stands and walks in MuJoCo viewer
- Brain activity drives locomotion (forward/backward/turn)
- Sensory feedback loop closed (gustatory zones, olfactory sources affect behavior)
- No crash/NaN over 60s simulation

---

## Removed vs. Kept

### Removed (fly-specific)
- `flight.py` — flight forces on thorax
- `vocalization.py` — wing song
- `looming_arena.py` — looming ball visual stimulus
- `procedural_arena.py` — fly-specific arena
- `flygym` dependency (all imports)
- `dm_control` dependency (MJCF merging in two_flies.py)
- `HybridTurningController`, `PreprogrammedSteps` (flygym classes)
- Proboscis joint
- Per-eye T2 threat bias (replaced by simpler escape logic)

### Unchanged (brain logic)
- `code/run_pytorch.py` — LIF model
- `brain_body_bridge.py` — BrainEngine, DNRateDecoder
- `visual_system.py` — photoreceptor mapping (if visual enabled)
- `somatosensory.py` — JO encoding (sensor array shape adapts)
- `gustatory.py` — GRN encoding (end_effector count adapts)
- `olfactory.py` — ORN encoding
- `consciousness.py` — spike readout
- `brain_monitor.py` — spike visualization

---

## Implementation Notes (2026-09-13)

### Go2 MJCF Model

- Actuators are position-type (trntype=0) with gain=[1, 0, ...] — too weak for direct standing
- **Solution**: PD torque control writing computed torques to `mj_data.ctrl`, matching the DDS bridge approach
- Gains: `kp=40, kd=1` produce stable standing at z≈0.273m

### Standing Pose

- Default all-zeros pose = extended legs (not standing)
- PD equilibrium: thigh≈0.3, knee≈-1.0, base z≈0.273m
- Joint ranges: hip [-1.05, 1.05], thigh_F [-1.57, 3.49], thigh_R [-0.52, 4.54], knee [-2.72, -0.84]

### Locomotion Direction

- **Thigh NEGATIVE = robot moves FORWARD** (empirically verified)
- Push amplitude: -0.5 rad on thigh during stance = ~0.15 m/s forward
- Foot lift: -0.35 rad knee bend on opposite diagonal pair

### CPG Evolution

Three iterations needed to find working pattern:
1. Sinusoidal oscillation → no net motion (symmetric push/cancel)
2. Sinusoidal with trot phase → backward motion (wrong push direction)
3. **Absolute push with opposite-pair foot lift → forward motion ✅**

### Brain-Body Integration Results

- DN forward rate: 0.625-0.745 (P9 stimulation at 100 Hz)
- Drive output: [~1.0, ~0.3] (forward + slight right bias)
- Forward displacement: ~0.3m over 2s simulation
- Realtime ratio: ~60x on CPU (GPU expected ~5x)
- Hebbian plasticity: active on 15M synapses

### Key Bug Fixed

DN spikes were being lost because `brain.get_dn_spikes()` was called only after the 10th substep. Fixed by accumulating spikes across all BRAIN_RATIO substeps with max-pooling.

### Viewer

Must use `mjpython` on macOS (MuJoCo requirement):
```bash
/Users/dan/miniconda3/envs/fly/bin/mjpython fly_embodied_unitree.py
```