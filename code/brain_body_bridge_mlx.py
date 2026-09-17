"""
brain_body_bridge_mlx.py — MLX backend for the fly-brain BrainEngine.

Provides MlxBrainEngine with the same interface as brain_body_bridge.BrainEngine,
but running the spiking model on Apple Silicon Metal via MLX (no PyTorch, no
mlx-sparse). The connectome recurrent matmul uses a scatter-add over COO edges,
which runs ~8× faster than the PyTorch CPU sparse CSR matmul.

Interface-compatible with BrainEngine:
  - flyid2i / i2flyid / num_neurons
  - set_stimulus / set_visual_rates / set_sensory_rates
  - step / get_dn_spikes / register_population / get_population_spikes
  - Hebbian plasticity (functional weight updates)
"""

import numpy as np
import mlx.core as mx
from pathlib import Path

from run_mlx import MlxModel, MODEL_PARAMS, DT, get_hash_tables, load_weights

# Reuse neuron IDs and stimulus definitions from brain_body_bridge (PyTorch
# import is harmless — torch is installed; we only read its constants).
from brain_body_bridge import (
    DN_NEURONS, STIMULI, DN_GROUPS,
    HEBB_BATCH, HEBB_ETA, HEBB_ALPHA, PLASTIC_PATH,
)


class MlxBrainEngine:
    """Wraps the MLX spiking model for stepwise execution on Metal."""

    def __init__(self, plastic_path=None):
        self.dt = DT
        self._plastic_path = Path(plastic_path) if plastic_path else PLASTIC_PATH

        data_dir = Path(__file__).resolve().parent.parent / 'data'
        comp_path = data_dir / '2025_Completeness_783.csv'
        conn_path = data_dir / '2025_Connectivity_783.parquet'

        # FlyWire ID <-> tensor index mappings
        self.flyid2i, self.i2flyid = get_hash_tables(str(comp_path))
        self.num_neurons = len(self.flyid2i)

        # Load connectome edges + build model
        _, row_idx, col_idx, val_np, pre_indptr = load_weights(
            str(conn_path), str(comp_path))
        self.model = MlxModel(self.num_neurons, DT, MODEL_PARAMS,
                              row_idx, col_idx, val_np, pre_indptr)

        # Initialize neural state
        self.state = self.model.state_init()

        # Input rate vector
        self.rates = mx.zeros((self.num_neurons,), dtype=mx.float32)

        # DN neuron indices
        self.dn_indices = {
            name: self.flyid2i[flyid]
            for name, flyid in DN_NEURONS.items() if flyid in self.flyid2i
        }

        # Stimulus neuron indices
        self.stim_indices = {
            stim_name: [self.flyid2i[nid] for nid in stim_info['neurons']
                        if nid in self.flyid2i]
            for stim_name, stim_info in STIMULI.items()
        }

        self.populations = {}

        print(f"[MlxBrainEngine] {self.num_neurons} neurons on Metal (MLX)")
        print(f"[MlxBrainEngine] DN neurons mapped: "
              f"{len(self.dn_indices)}/{len(DN_NEURONS)}")
        for s, idx in self.stim_indices.items():
            print(f"  '{s}': {len(idx)}/{len(STIMULI[s]['neurons'])} neurons")

        self._init_plasticity()

    # ── Hebbian Plasticity ──────────────────────────────────────────────

    def _init_plasticity(self):
        val = self.model.val  # (nnz,)
        self._sign_mask = mx.sign(val)
        self._abs_orig = mx.abs(val)
        max_mag = 3.0 * self._abs_orig
        self._clamp_min = mx.where(self._sign_mask < 0, -max_mag,
                                   mx.zeros_like(max_mag))
        self._clamp_max = mx.where(self._sign_mask > 0, max_mag,
                                   mx.zeros_like(max_mag))
        self._spike_acc = mx.zeros((self.num_neurons,), dtype=mx.float32)
        self._hebb_count = 0
        self._decay_count = 0
        self._decay_every = 100  # apply multiplicative decay every N hebb updates

        if self._plastic_path.exists():
            saved = np.load(self._plastic_path) if self._plastic_path.suffix == '.npy' \
                else None
            if saved is not None and saved.shape == np.array(val).shape:
                self.model.val = mx.array(saved.astype(np.float32))
                self._sign_mask = mx.sign(self.model.val)
                print(f"[MlxBrainEngine] Loaded plastic weights from {self._plastic_path}")

        print(f"[MlxBrainEngine] Hebbian plasticity active: "
              f"{len(val)} synapses")

    def _hebb_update(self):
        """Event-driven Hebbian potentiation + lazy multiplicative decay.

        Potentiation is applied only to synapses whose PRE neuron was active
        (avg>0) in the last HEBB_BATCH steps — under sparse activity this is
        a handful of edges instead of all 15M. The multiplicative decay
        (-HEBB_ALPHA·val) is folded into a lazy bulk multiply every
        _decay_every updates, since HEBB_ALPHA=1e-7 is negligible short-term.
        """
        avg_np = np.array(self._spike_acc / HEBB_BATCH).astype(np.float32)
        self._spike_acc = mx.zeros((self.num_neurons,), dtype=mx.float32)

        # --- Event-driven potentiation (co-active synapses only) ---
        active = np.nonzero(avg_np > 0)[0]
        if active.size > 0:
            ip = self.model.pre_indptr
            starts = ip[active]
            ends = ip[active + 1]
            flat = np.concatenate([
                np.arange(starts[k], ends[k], dtype=np.int32)
                for k in range(active.size)
            ])
            if flat.size > 0:
                flat_mx = mx.array(flat)
                edge_post_np = np.array(mx.take(self.model.row_idx, flat_mx)).astype(np.int32)
                edge_val = mx.take(self.model.val, flat_mx)
                edge_sign = mx.take(self._sign_mask, flat_mx)
                pre_avg = np.repeat(avg_np[active], ends - starts).astype(np.float32)
                post_avg = avg_np[edge_post_np].astype(np.float32)

                dW = (HEBB_ETA * mx.array(pre_avg) * mx.array(post_avg)
                      * edge_sign)
                edge_min = mx.take(self._clamp_min, flat_mx)
                edge_max = mx.take(self._clamp_max, flat_mx)
                new_edge = mx.clip(edge_val + dW, edge_min, edge_max)
                delta = new_edge - edge_val
                self.model.val = self.model.val.at[flat_mx].add(delta)

        # --- Lazy multiplicative decay ---
        self._decay_count += 1
        if self._decay_count >= self._decay_every:
            factor = 1.0 - HEBB_ALPHA * self._decay_every
            self.model.val = self.model.val * float(factor)
            self._decay_count = 0

    def save_plastic_weights(self):
        np.save(self._plastic_path.with_suffix('.npy'),
                np.array(self.model.val).astype(np.float32))
        print(f"[MlxBrainEngine] Saved plastic weights to "
              f"{self._plastic_path.with_suffix('.npy')}")

    # ── Input setters ───────────────────────────────────────────────────

    def set_stimulus(self, stim_name):
        self.rates = mx.zeros((self.num_neurons,), dtype=mx.float32)
        if stim_name and stim_name in STIMULI:
            idx = self.stim_indices.get(stim_name, [])
            if idx:
                self.rates = self.rates.at[mx.array(idx, dtype=mx.int32)].add(
                    mx.array([STIMULI[stim_name]['rate']], dtype=mx.float32))

    def set_visual_rates(self, photo_indices, photo_rates):
        if photo_indices is None or len(photo_indices) == 0:
            return
        idx = mx.array(np.asarray(photo_indices, dtype=np.int32))
        val = mx.array(np.asarray(photo_rates, dtype=np.float32))
        self.rates = self.rates.at[idx].add(val)

    def set_sensory_rates(self, indices, rates):
        if indices is None or len(indices) == 0:
            return
        idx = mx.array(np.asarray(indices, dtype=np.int32))
        new_rates = mx.array(np.asarray(rates, dtype=np.float32))
        current = mx.take(self.rates, idx)
        self.rates = self.rates.at[idx].add(mx.maximum(0, new_rates - current))

    # ── Execution ───────────────────────────────────────────────────────

    def step(self):
        """Advance brain by one timestep (0.1 ms). Returns spike vector (n,)."""
        cond, dbuf, spk, v, ref = self.state
        self.state = self.model.step(self.rates, cond, dbuf, spk, v, ref)
        spikes = self.state[2]
        self._spike_acc = self._spike_acc + spikes
        self._hebb_count += 1
        if self._hebb_count >= HEBB_BATCH:
            self._hebb_update()
            self._hebb_count = 0
        return spikes

    def get_dn_spikes(self):
        spk = np.array(self.state[2]).astype(np.float32)  # materialize once
        return {name: float(spk[idx]) for name, idx in self.dn_indices.items()}

    def register_population(self, name, tensor_indices):
        self.populations[name] = np.asarray(tensor_indices, dtype=np.int32)
        print(f"[MlxBrainEngine] Population '{name}': "
              f"{len(tensor_indices)} neurons")

    def get_population_spikes(self):
        spk = self.state[2]
        result = {}
        for name, indices in self.populations.items():
            result[name] = float(mx.mean(mx.take(spk, mx.array(indices))))
        return result

    def get_spike_count(self, indices):
        """Total spike count for given tensor indices (int)."""
        spk = self.state[2]
        idx = mx.array(np.asarray(indices, dtype=np.int32))
        return int(np.array(mx.sum(mx.take(spk, idx))))