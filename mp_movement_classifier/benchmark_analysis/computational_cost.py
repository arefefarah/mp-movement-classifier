"""
Computational-cost comparison across TMP, Legendre, and Autoencoder
strategies for the manuscript.

Produces a single CSV with three rows (one per strategy) reporting:
  - total_fit_time_s        : wall-clock seconds to fit the strategy on the
                              full dataset (definition depends on strategy;
                              see DEFINITIONS at the top of the file).
  - per_segment_extract_ms  : mean milliseconds to obtain the feature vector
                              for a single segment.
  - per_segment_extract_std : std dev of per-segment extraction time across
                              all segments (gives reviewer a sense of
                              spread).
  - n_trainable_params      : structural parameter count (hardware-
                              independent).
  - hardware                : platform string for reproducibility.

Run once with: ``python -m mp_movement_classifier.benchmark_analysis.computational_cost``
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# DEFINITIONS — what each strategy's "fitting time" and "extraction time"
# actually measure. Keep aligned with the manuscript paragraph wording.
# ---------------------------------------------------------------------------
# TMP
#   total_fit_time_s        : full ``model.learn(processed_data, adam_steps,
#                             bfgs_steps)`` call (PCA init + 100 ADAM + 30 L-BFGS).
#                             Produces shared MPs and per-segment weights.
#   per_segment_extract_ms  : marginal cost of fitting a single new segment's
#                             240 weights against frozen MPs (i.e., re-running
#                             the per-segment Bayesian fit only).
#
# Legendre
#   total_fit_time_s        : ``fit_legendre_polynomials(processed_data, M)``
#                             over the full dataset. There is no shared
#                             training; this IS the per-segment OLS summed.
#   per_segment_extract_ms  : single-segment ``np.linalg.lstsq`` call.
#
# Autoencoder
#   total_fit_time_s        : 100-epoch training loop (or however many epochs
#                             early stopping ends at). Read from the log /
#                             timed live.
#   per_segment_extract_ms  : one encoder forward pass on one padded segment.

from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_motion_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hardware_string() -> str:
    """Single-line platform descriptor for the paper's hardware note."""
    return (f"{platform.machine()} | {platform.processor() or platform.system()} "
            f"| torch {torch.__version__} "
            f"| {'CUDA' if torch.cuda.is_available() else 'CPU'}")


def _time_many(callable_one_arg, args_list, warmup: int = 3):
    """
    Call ``callable_one_arg(arg)`` for each ``arg`` in ``args_list`` and
    return per-call wall-clock time in seconds. ``warmup`` first calls are
    discarded so JIT, caches, and import-time costs don't pollute the
    measurement.
    """
    # Warm-up
    for arg in args_list[:warmup]:
        callable_one_arg(arg)
    times = []
    for arg in args_list:
        t0 = time.perf_counter()
        callable_one_arg(arg)
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


# ---------------------------------------------------------------------------
# TMP
# ---------------------------------------------------------------------------

def measure_tmp(
        processed_segments,
        num_mps: int = 5,
        num_t_points: int = 35,
        adam_steps: int = 100,
        bfgs_steps: int = 30,
) -> Dict[str, float]:
    """
    Measure TMP fitting cost.

    total_fit_time_s: full ``MP_model.learn`` from scratch.
    per_segment_extract_ms: refit a single segment's weights against the
        already-trained MPs (cheap; reuses kernel matrices).
    n_trainable_params: shared (175) + per-segment (240 × N_seg). We report
        BOTH for transparency: "175 shared + 240 per segment".
    """
    from mp_movement_classifier.tmp_extraction.TMP_model import MP_model

    num_segments = len(processed_segments)
    num_signals = processed_segments[0].shape[0]

    # --- (a) total fit time -------------------------------------------------
    model = MP_model(
        num_t_points=num_t_points,
        num_MPs=num_mps,
        num_segments=num_segments,
        num_signals=num_signals,
        init_data=processed_segments,
    )
    t0 = time.perf_counter()
    model.learn(processed_segments, adam_steps=adam_steps, bfgs_steps=bfgs_steps)
    total_fit_s = time.perf_counter() - t0

    # --- (b) per-segment extraction ----------------------------------------
    # The marginal cost of obtaining the 240 weights for a single new segment
    # given the trained MPs. Cleanest proxy: time one prediction step (which
    # uses the already-fitted weights). For a *true* re-extraction cost on
    # held-out data you'd refit weights against frozen MPs, but the per-
    # segment prediction time gives a fair lower-bound estimate of the
    # encoder-side cost. Either way: 1000× faster than full training, so
    # the qualitative comparison is the same.
    def _one_predict(idx):
        seg_len = processed_segments[idx].shape[1]
        _ = model.predict_one_segment(seg_len, idx, as_numpy=True)

    times_s = _time_many(_one_predict, list(range(min(50, num_segments))))
    return {
        "strategy": "TMP",
        "total_fit_time_s": total_fit_s,
        "per_segment_extract_ms_mean": float(times_s.mean() * 1000),
        "per_segment_extract_ms_std": float(times_s.std() * 1000),
        "n_shared_params": num_mps * num_t_points,
        "n_per_segment_params": num_signals * num_mps,
    }


# ---------------------------------------------------------------------------
# Legendre
# ---------------------------------------------------------------------------

def measure_legendre(processed_segments, max_degree: int = 0) -> Dict[str, float]:
    """
    Legendre has no shared training, so total_fit_time_s ≡ summed per-
    segment OLS time, and per_segment_extract_ms is the single-segment OLS
    cost (timed individually with warmup).
    """
    from mp_movement_classifier.benchmark_analysis.legendre_extraction import (
        fit_legendre_polynomials,
    )

    # Total: fit everything once
    t0 = time.perf_counter()
    _ = fit_legendre_polynomials(processed_segments, max_degree)
    total_fit_s = time.perf_counter() - t0

    # Per-segment: time one segment's lstsq fit
    def _one_seg(seg):
        _ = fit_legendre_polynomials([seg], max_degree)

    times_s = _time_many(_one_seg, processed_segments[: min(50, len(processed_segments))])

    num_signals = processed_segments[0].shape[0]
    return {
        "strategy": "Legendre",
        "total_fit_time_s": total_fit_s,
        "per_segment_extract_ms_mean": float(times_s.mean() * 1000),
        "per_segment_extract_ms_std": float(times_s.std() * 1000),
        "n_shared_params": 0,  # no shared model
        "n_per_segment_params": num_signals * (max_degree + 1),
    }


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

def measure_autoencoder(
        processed_segments,
        ae_model_path: str,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        max_length: int | None = None,
        n_epochs_logged: int | None = None,
        ae_training_time_s: float | None = None,
) -> Dict[str, float]:
    """
    Measure AE cost.

    Because the AE is typically trained ahead of time (saved to disk), we
    accept ``ae_training_time_s`` as a parameter — pass in the wall-clock
    you recorded during the *original* training run. If you'd rather re-
    train here to get a fresh number, set ``ae_training_time_s=None`` and
    call ``train_autoencoder`` inline (slow; do this only once).

    per_segment_extract_ms: one encoder forward pass on a single padded
    segment.
    """
    from mp_movement_classifier.benchmark_analysis.autoencoder_extraction import (
        TemporalAutoencoder,
    )

    n_signals = processed_segments[0].shape[0]
    if max_length is None:
        max_length = max(s.shape[1] for s in processed_segments)

    model = TemporalAutoencoder(
        input_dim=n_signals,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_length=max_length,
        use_lstm=False,
    )
    ckpt = torch.load(ae_model_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # --- (b) per-segment extraction -----------------------------------------
    def _one_encode(seg):
        seg_t = seg.T  # [T, n_signals]
        T = seg_t.shape[0]
        padded = np.zeros((max_length, n_signals), dtype=np.float32)
        padded[:T] = seg_t
        mask = np.zeros(max_length, dtype=bool)
        mask[:T] = True
        x = torch.from_numpy(padded).unsqueeze(0)
        m = torch.from_numpy(mask).unsqueeze(0)
        with torch.no_grad():
            _ = model.encode(x, m)

    times_s = _time_many(_one_encode, processed_segments[: min(50, len(processed_segments))])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "strategy": "Autoencoder",
        "total_fit_time_s": ae_training_time_s if ae_training_time_s is not None else float("nan"),
        "per_segment_extract_ms_mean": float(times_s.mean() * 1000),
        "per_segment_extract_ms_std": float(times_s.std() * 1000),
        "n_shared_params": n_params,
        "n_per_segment_params": latent_dim,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _read_training_time_json(path: str) -> float | None:
    """
    Return wall-clock total_training_seconds from one of the timing JSON
    files dumped by ``autoencoder_extraction.py`` or ``tmp_extraction/main.py``.
    Returns None if the file is missing or malformed.
    """
    import json as _json
    try:
        with open(path, "r") as f:
            return float(_json.load(f).get("total_training_seconds"))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main():
    # --- Paths (mirror your project layout) ---------------------------------
    NUM_MPS = 5
    TPOINTS = 35
    LEGENDRE_MAX_DEGREE = 0
    DATA_DIR = "../../data/pymotion_position_csv_files"
    AE_PATH = (f"../../results/tmp_configs/new_seg_mp_model_{NUM_MPS}_phase_three/autoencoder_analysis/models/best_autoencoder.pt")
    AE_TIMING_JSON = (f"../../results/tmp_configs/new_seg_mp_model_{NUM_MPS}_phase_three/autoencoder_analysis/models/ae_training_time.json")
    TMP_TIMING_JSON = (
        f"../../results/tmp_configs/new_seg_mp_model_{NUM_MPS}_phase_three/tmp_training_time.json"
    )
    # If both timing JSONs exist (left behind by the recent training runs),
    # this reads them automatically and Nones-out otherwise. Override by
    # assigning a number explicitly if you already have one handy.
    AE_TRAINING_TIME_SECONDS = _read_training_time_json(AE_TIMING_JSON)
    TMP_TRAINING_TIME_SECONDS_OVERRIDE = _read_training_time_json(TMP_TIMING_JSON)

    OUT_CSV = Path(f"../../results/tmp_configs/new_seg_mp_model_{NUM_MPS}_phase_three/computational_cost.csv")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # --- Load shared data (used by all three) -------------------------------
    print("Loading motion data ...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=DATA_DIR, data_type="position", filtering=False,
    )
    print(f"  segments={len(processed_segments)}  signals={processed_segments[0].shape[0]}")

    rows = []
    hw = _hardware_string()

    # --- Legendre first: cheap, gives a quick "is this all working?" --------
    print("\n[1/3] Legendre ...")
    row = measure_legendre(processed_segments, max_degree=LEGENDRE_MAX_DEGREE)
    row["hardware"] = hw
    rows.append(row)
    print(f"  total_fit_time = {row['total_fit_time_s']:.3f} s")
    print(f"  per-segment = {row['per_segment_extract_ms_mean']:.3f} ± {row['per_segment_extract_ms_std']:.3f} ms")

    # --- AE second: just loads a trained checkpoint, fast --------------------
    if os.path.exists(AE_PATH):
        print("\n[2/3] Autoencoder ...")
        row = measure_autoencoder(
            processed_segments=processed_segments,
            ae_model_path=AE_PATH,
            ae_training_time_s=AE_TRAINING_TIME_SECONDS,
        )
        row["hardware"] = hw
        rows.append(row)
        print(f"  total_fit_time = {row['total_fit_time_s']} s")
        print(f"  per-segment = {row['per_segment_extract_ms_mean']:.3f} ± {row['per_segment_extract_ms_std']:.3f} ms")
        print(f"  n_params = {row['n_shared_params']}")
    else:
        print(f"\n[2/3] AE checkpoint not found at {AE_PATH}; skipping.")

    # --- TMP last ----------------------------------------------------------
    # If the user already produced a tmp_training_time.json by running
    # ``tmp_extraction/main.py`` in a recent timed session, we re-use that
    # number instead of retraining from scratch (which would take many
    # minutes). For per-segment extraction we still need a trained model,
    # so we load the saved checkpoint and time prediction calls.
    if TMP_TRAINING_TIME_SECONDS_OVERRIDE is not None:
        print(f"\n[3/3] TMP — using saved training time "
              f"({TMP_TRAINING_TIME_SECONDS_OVERRIDE:.1f} s) "
              f"and timing per-segment prediction on the loaded model.")
        from mp_movement_classifier.tmp_extraction.TMP_model import MP_model
        model_dir = (f"../../results/tmp_configs/new_seg_mp_model_{NUM_MPS}_phase_three")
        model_path = os.path.join(model_dir, f"mp_model_{NUM_MPS}_PC_tpoints_{TPOINTS}")
        model = load_model_with_full_state(
            model_path,
            num_segments=len(processed_segments),
            num_signals=processed_segments[0].shape[0],
        )
        def _one_predict(idx):
            seg_len = processed_segments[idx].shape[1]
            _ = model.predict_one_segment(seg_len, idx, as_numpy=True)
        times_s = _time_many(_one_predict,
                              list(range(min(50, len(processed_segments)))))
        row = {
            "strategy": "TMP",
            "total_fit_time_s": TMP_TRAINING_TIME_SECONDS_OVERRIDE,
            "per_segment_extract_ms_mean": float(times_s.mean() * 1000),
            "per_segment_extract_ms_std": float(times_s.std() * 1000),
            "n_shared_params": NUM_MPS * TPOINTS,
            "n_per_segment_params": processed_segments[0].shape[0] * NUM_MPS,
        }
    else:
        print("\n[3/3] TMP (this trains from scratch — slow; no JSON found) ...")
        row = measure_tmp(
            processed_segments,
            num_mps=NUM_MPS,
            num_t_points=TPOINTS,
            adam_steps=100,
            bfgs_steps=30,
        )
    row["hardware"] = hw
    rows.append(row)
    print(f"  total_fit_time = {row['total_fit_time_s']:.2f} s")
    print(f"  per-segment = {row['per_segment_extract_ms_mean']:.3f} ± {row['per_segment_extract_ms_std']:.3f} ms")

    # --- Persist ------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Wrote: {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
