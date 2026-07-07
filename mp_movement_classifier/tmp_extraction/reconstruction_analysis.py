"""
Reconstruction-quality analysis for movement-primitive strategies.
=================================================================

Static, manuscript-grade quantification of how well each feature-extraction
strategy (TMP / Legendre / Autoencoder) reconstructs the original motion
trajectories. Replaces the qualitative GIFs from
``recontruction_visulaization.py`` with reviewer-expected numeric metrics
and static figures.

Metrics (computed per segment, then aggregated)
-----------------------------------------------
* **VAF** — Variance Accounted For. Field-standard for MP papers
  (d'Avella & Bizzi 2005, Tresch et al. 2006, Chiovetto et al. 2013).
  VAF = 1 − SSres / SStot, in (−∞, 1], higher = better.
* **MPJPE** — Mean Per-Joint Position Error. Standard in mocap-reconstruction
  papers (Ionescu et al. 2014, Pavllo et al. 2019). Average Euclidean
  distance per joint, averaged across joints and timesteps. Units match the
  input (mm or normalized — reported in input units).
* **Velocity-RMSE** — RMSE in the velocity domain (np.gradient on time
  axis). Position-only metrics are dominated by posture; vRMSE catches
  whether MP dynamics actually fit.
* **Pearson r** — mean across-channel Pearson correlation between real
  and reconstructed trajectories. Scale-/offset-invariant complement to
  RMSE (used in d'Avella et al. 2003).

Reconstruction protocol
-----------------------
Each strategy's *own learned per-segment features* are used to reconstruct
that same segment (no train/test split — this is the standard reconstruction-
fidelity protocol from the MP literature; classification quality is
evaluated separately by ``run_classification_pipeline``). The reconstruction
is then compared to the ground-truth input segment of identical length.

For AE specifically, padding is reversed via the mask so metrics are
computed only on valid timesteps.

Outputs (under ``<save_dir>/``)
-------------------------------
* ``reconstruction_metrics_per_segment.csv`` — one row per (strategy,
  segment): vaf, mpjpe, velocity_rmse, pearson_r, motion_id, motion_name,
  segment_length.
* ``reconstruction_metrics_summary.csv`` — one row per strategy:
  mean ± std for each metric across all segments.
* ``reconstruction_metrics_per_class.csv`` — one row per (strategy, class):
  mean ± std within each motion class.
* ``vaf_by_strategy_bar.{png,svg}`` — mean VAF (±std) per strategy.
* ``vaf_per_class.{png,svg}`` — per-class VAF, grouped bars (one cluster
  per movement, one bar color per strategy).
* ``mpjpe_per_class.{png,svg}`` — per-class MPJPE, grouped bars.
* ``trajectory_overlay_<motion>.{png,svg}`` — real vs reconstructed
  trajectory for representative joints, one figure per shown motion.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_motion_data,
)
from mp_movement_classifier.benchmark_analysis.legendre_extraction import (
    fit_legendre_polynomials,
    reconstruct_from_coefficients,
)


# =============================================================================
# Reconstruction backends (one function per strategy)
# =============================================================================

def reconstruct_tmp(
        model,
        processed_segments: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Reconstruct every segment using its own learned TMP weights and the
    shared movement primitives.

    Each segment is returned with shape ``[n_signals, T_i]`` (matching the
    input convention).
    """
    segment_lengths = np.array([segment.shape[1] for segment in processed_segments], dtype=int)
    return model.predict(segment_lengths, as_numpy=True)


def reconstruct_legendre(
        processed_segments: List[np.ndarray],
        max_degree: int,
) -> List[np.ndarray]:
    """
    Per-segment Legendre reconstruction: fit each segment's coefficients on
    the fly with ``fit_legendre_polynomials``, then project them back through
    the basis at the segment's original length.

    Returns segments in the same ``[n_signals, T_i]`` convention as the input.
    """
    coefficients = fit_legendre_polynomials(processed_segments, max_degree)
    out: List[np.ndarray] = []
    for i, seg in enumerate(processed_segments):
        T = seg.shape[1]
        out.append(reconstruct_from_coefficients(coefficients[i], T, max_degree))
    return out


def reconstruct_autoencoder(
        model,
        processed_segments: List[np.ndarray],
        device: str = "cpu",
        max_length: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Per-segment autoencoder reconstruction.

    The AE was trained on segments shaped ``[T, n_signals]`` and padded to a
    fixed ``max_length``. We replicate that exactly here, then slice the
    decoder output back to each segment's true length and transpose so the
    returned segments match the canonical ``[n_signals, T_i]`` convention.
    """
    if max_length is None:
        max_length = max(seg.shape[1] for seg in processed_segments)

    n_signals = processed_segments[0].shape[0]

    model.eval()
    out: List[np.ndarray] = []
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for seg in processed_segments:
            seg_t = seg.T  # → [T, n_signals]
            T = seg_t.shape[0]

            padded = np.zeros((max_length, n_signals), dtype=np.float32)
            padded[:T] = seg_t
            mask = np.zeros(max_length, dtype=bool)
            mask[:T] = True

            x = torch.from_numpy(padded).to(device=device, dtype=model_dtype).unsqueeze(0)
            m = torch.from_numpy(mask).to(device=device).unsqueeze(0)

            recon, _ = model(x, m)            # [1, max_length, n_signals]
            recon_np = recon.cpu().numpy()[0, :T, :]  # crop to valid length
            out.append(recon_np.T)            # → [n_signals, T_i]

    return out


# =============================================================================
# Metric primitives — operate on a single (real, reconstructed) segment pair
# =============================================================================

def vaf(real: np.ndarray, recon: np.ndarray) -> float:
    """
    VAF = 1 − Σ(real − recon)² / Σ(real − mean(real))²

    Computed over all channels and timesteps. Returns NaN for degenerate
    (zero-variance) segments instead of raising.
    """
    res = (real - recon) ** 2
    var = (real - real.mean(axis=1, keepdims=True)) ** 2
    denom = float(var.sum())
    if denom <= 1e-20:
        return float("nan")
    return 1.0 - float(res.sum()) / denom


def mpjpe(real: np.ndarray, recon: np.ndarray, n_joints: int = 16) -> float:
    """
    Mean Per-Joint Position Error.

    Real/recon shape ``[n_signals = 3*n_joints, T]``. Reshapes to
    ``[T, J, 3]`` and reports the mean Euclidean distance per joint
    averaged over joints and time. Units = input units (mm if the data is in mm).
    """
    if real.shape[0] != 3 * n_joints:
        # Fall back to channel-wise RMS distance (no spatial grouping) when
        # the input isn't shaped like 3D joint coordinates.
        return float(np.sqrt(((real - recon) ** 2).mean()))
    real_3d = real.T.reshape(-1, n_joints, 3)
    recon_3d = recon.T.reshape(-1, n_joints, 3)
    per_joint_dist = np.linalg.norm(real_3d - recon_3d, axis=2)  # [T, J]
    return float(per_joint_dist.mean())


def velocity_rmse(real: np.ndarray, recon: np.ndarray) -> float:
    """
    RMSE between the time-derivatives of real and reconstructed signals.

    Crucial for MP-style models: position RMSE can look fine even when the
    *dynamics* are smeared out by too few primitives.
    """
    if real.shape[1] < 2:
        return float("nan")
    dreal = np.gradient(real, axis=1)
    drecon = np.gradient(recon, axis=1)
    return float(np.sqrt(((dreal - drecon) ** 2).mean()))


def mean_pearson_r(real: np.ndarray, recon: np.ndarray) -> float:
    """
    Channel-wise Pearson correlation between real and reconstructed time
    series, averaged across channels. Channels with degenerate variance
    on either side are skipped.
    """
    rs: List[float] = []
    for ch in range(real.shape[0]):
        a, b = real[ch], recon[ch]
        if a.std() > 1e-12 and b.std() > 1e-12:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")


# =============================================================================
# Aggregation
# =============================================================================

@dataclass
class StrategyResult:
    name: str
    reconstructed: List[np.ndarray]
    per_segment: pd.DataFrame  # columns: vaf, mpjpe, velocity_rmse, pearson_r, motion_id, motion_name, segment_length


def compute_per_segment_metrics(
        strategy_name: str,
        real_segments: List[np.ndarray],
        recon_segments: List[np.ndarray],
        motion_ids: Sequence[int],
        motion_id_to_name: Dict[int, str],
        n_joints: int,
) -> pd.DataFrame:
    rows = []
    for i, (real, recon) in enumerate(zip(real_segments, recon_segments)):
        mid = int(motion_ids[i])
        rows.append({
            "strategy": strategy_name,
            "segment_idx": i,
            "motion_id": mid,
            "motion_name": motion_id_to_name.get(mid, str(mid)),
            "segment_length": real.shape[1],
            "vaf": vaf(real, recon),
            "mpjpe": mpjpe(real, recon, n_joints=n_joints),
            "velocity_rmse": velocity_rmse(real, recon),
            "pearson_r": mean_pearson_r(real, recon),
        })
    return pd.DataFrame(rows)


def _q25(s):  # named so pandas .agg keeps a clean column name
    return s.quantile(0.25)


def _q75(s):
    return s.quantile(0.75)


def summarize_per_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per strategy with both *mean ± std* and *median + IQR* for each
    metric. Mean is dominated by heavy-tailed outliers (e.g. AE on
    near-stationary classes can give VAF ≪ 0); median + IQR are robust.
    Both are reported so the manuscript can use whichever fits the section.
    """
    agg = df.groupby("strategy").agg(
        n_segments=("segment_idx", "count"),
        vaf_mean=("vaf", "mean"), vaf_std=("vaf", "std"),
        vaf_median=("vaf", "median"), vaf_q25=("vaf", _q25), vaf_q75=("vaf", _q75),
        mpjpe_mean=("mpjpe", "mean"), mpjpe_std=("mpjpe", "std"),
        mpjpe_median=("mpjpe", "median"), mpjpe_q25=("mpjpe", _q25), mpjpe_q75=("mpjpe", _q75),
        vrmse_mean=("velocity_rmse", "mean"), vrmse_std=("velocity_rmse", "std"),
        vrmse_median=("velocity_rmse", "median"), vrmse_q25=("velocity_rmse", _q25), vrmse_q75=("velocity_rmse", _q75),
        pearson_mean=("pearson_r", "mean"), pearson_std=("pearson_r", "std"),
        pearson_median=("pearson_r", "median"), pearson_q25=("pearson_r", _q25), pearson_q75=("pearson_r", _q75),
    ).reset_index()
    return agg


def summarize_per_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (strategy, motion class) with both *mean ± std* and
    *median + IQR* for each metric (see ``summarize_per_strategy`` for why
    both).
    """
    agg = df.groupby(["strategy", "motion_id", "motion_name"]).agg(
        n_segments=("segment_idx", "count"),
        vaf_mean=("vaf", "mean"), vaf_std=("vaf", "std"),
        vaf_median=("vaf", "median"), vaf_q25=("vaf", _q25), vaf_q75=("vaf", _q75),
        mpjpe_mean=("mpjpe", "mean"), mpjpe_std=("mpjpe", "std"),
        mpjpe_median=("mpjpe", "median"), mpjpe_q25=("mpjpe", _q25), mpjpe_q75=("mpjpe", _q75),
        vrmse_mean=("velocity_rmse", "mean"), vrmse_std=("velocity_rmse", "std"),
        vrmse_median=("velocity_rmse", "median"), vrmse_q25=("velocity_rmse", _q25), vrmse_q75=("velocity_rmse", _q75),
        pearson_mean=("pearson_r", "mean"), pearson_std=("pearson_r", "std"),
        pearson_median=("pearson_r", "median"), pearson_q25=("pearson_r", _q25), pearson_q75=("pearson_r", _q75),
    ).reset_index()
    return agg


# =============================================================================
# Figures
# =============================================================================

# Consistent palette across all figures so a reviewer sees the same color
# mean the same strategy in every panel of the paper.
# Strategy palette — kept in lockstep with
# ``run_classification_all_models._plot_cross_validation_comparison`` so the
# manuscript shows the same colour for the same strategy across the
# classification CV figure, the PCA/LDA combined figures, and the
# reconstruction-analysis figures. Update both places together if the
# project-wide palette ever changes.
STRATEGY_COLORS = {
    "TMP":         "#3498db",   # sky blue   (was #1f77b4)
    "Legendre":    "#2ecc71",   # emerald    (was #2ca02c)
    "Autoencoder": "#e74c3c",   # coral red  (was #ff7f0e)
}

# All text in the reconstruction figures uses these sizes (in points).
# Minimum is 8 so labels remain legible after the figures are scaled down
# into the two-column manuscript layout. Kept in one place so any future
# adjustment is a single-line change.
FONT = dict(
    tick=9,        # x/y tick labels
    label=10,      # axis labels (xlabel / ylabel)
    title=11,      # subplot/figure titles
    legend=9,      # legend body
    annot=8,       # in-axes secondary text
    suptitle=12,   # figure-level title (only used in trajectory overlay)
)


def _save_png_svg(path: Path) -> None:
    p = Path(path)
    plt.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(p.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def plot_summary_bars(summary: pd.DataFrame, save_dir: Path) -> None:
    """
    One-row, four-panel summary: VAF, MPJPE, velocity-RMSE, Pearson r,
    each as bar (mean) + error bar (std), with one bar per strategy.
    """
    metrics = [
        ("vaf_mean",     "vaf_std",     "VAF (higher better)",          False),
        ("mpjpe_mean",   "mpjpe_std",   "MPJPE (lower better)",         True),
        ("vrmse_mean",   "vrmse_std",   "Velocity-RMSE (lower better)", True),
        ("pearson_mean", "pearson_std", "Pearson r (higher better)",    False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    strategies = list(summary["strategy"])
    colors = [STRATEGY_COLORS.get(s, "gray") for s in strategies]

    for ax, (mcol, scol, title, _lower_is_better) in zip(axes, metrics):
        means = summary[mcol].values
        stds = summary[scol].values
        x = np.arange(len(strategies))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors,
               edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, fontsize=FONT["tick"])
        ax.set_title(title, fontsize=FONT["title"], fontweight="bold")
        ax.tick_params(axis="y", labelsize=FONT["tick"])
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save_png_svg(save_dir / "reconstruction_summary_bars")
    plt.close(fig)


def plot_per_class_bars(
        per_class: pd.DataFrame,
        metric_center: str,
        metric_low: Optional[str],
        metric_high: Optional[str],
        ylabel: str,
        title: str,
        save_dir: Path,
        file_stem: str,
        stat_kind: str = "mean",
        ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Grouped bar chart: one cluster per motion class, one bar per strategy.

    Parameters
    ----------
    metric_center : column name for the bar height (``..._mean`` or ``..._median``).
    metric_low, metric_high :
        For ``stat_kind="mean"``: ``metric_low`` should be the ``..._std``
        column and ``metric_high`` should be ``None`` — error bars are
        rendered symmetrically as ``±std``.
        For ``stat_kind="median"``: ``metric_low`` / ``metric_high`` should
        be the ``..._q25`` / ``..._q75`` columns — error bars are rendered
        as asymmetric IQR ``[median−q25, q75−median]``.
    ylim : optional (low, high) hard clip applied to the y-axis after
        autoscaling. Use this when a few catastrophic outliers (e.g. AE on
        near-stationary classes) would otherwise crush all readable values
        toward zero. The underlying data in the CSV is untouched.
    """
    strategies = sorted(per_class["strategy"].unique(),
                        key=lambda s: list(STRATEGY_COLORS).index(s)
                        if s in STRATEGY_COLORS else 99)

    # Direction depends on the metric: error metrics ascend (best at left),
    # quality metrics descend (best at left). We detect from the column stem.
    ascending = any(tag in metric_center for tag in ("mpjpe", "vrmse"))
    ref = (per_class[per_class["strategy"] == strategies[0]]
           .sort_values(metric_center, ascending=ascending))
    motion_order = ref["motion_name"].tolist()

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(motion_order)), 4.2))
    n_strats = len(strategies)
    bar_w = 0.8 / n_strats
    x = np.arange(len(motion_order))

    # Collect per-strategy bar geometry first so we can annotate clipped
    # bars in a second pass *after* matplotlib has settled the axis limits.
    bar_positions: List[Tuple[str, float, float]] = []  # (strategy, x_pos, center_value)

    for i, s in enumerate(strategies):
        sub = (per_class[per_class["strategy"] == s]
               .set_index("motion_name")
               .reindex(motion_order))
        center = sub[metric_center].values
        offset = (i - (n_strats - 1) / 2) * bar_w

        if stat_kind == "mean":
            yerr = sub[metric_low].values if metric_low else None
        else:  # median + IQR
            lo = sub[metric_low].values
            hi = sub[metric_high].values
            # Asymmetric error bars: [distance below, distance above]
            yerr = np.vstack([
                np.clip(center - lo, 0, None),
                np.clip(hi - center, 0, None),
            ])

        ax.bar(x + offset, center, width=bar_w,
               yerr=yerr, capsize=2,
               color=STRATEGY_COLORS.get(s, "gray"),
               edgecolor="black", linewidth=0.4, alpha=0.85, label=s)

        for xp, cv in zip(x + offset, center):
            bar_positions.append((s, float(xp), float(cv)))

    ax.set_xticks(x)
    ax.set_xticklabels(motion_order, rotation=45, ha="right",
                       fontsize=FONT["tick"])
    ax.set_ylabel(ylabel, fontsize=FONT["label"], fontweight="bold")
    ax.set_title(title, fontsize=FONT["title"], fontweight="bold")
    ax.tick_params(axis="y", labelsize=FONT["tick"])
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FONT["legend"], frameon=True, framealpha=0.9, loc="best")

    if ylim is not None:
        ax.set_ylim(*ylim)

        # Annotate any bar whose center falls below/above the clipped y-axis
        # with a small down/up arrow at the boundary plus the raw value, so a
        # reviewer can see at a glance "this bar is off-scale, the real value
        # is X." Underlying data in the CSV is unchanged.
        lo, hi = ylim
        span = hi - lo
        for s, xp, cv in bar_positions:
            if not np.isfinite(cv):
                continue
            if cv < lo:
                ax.annotate(
                    f"▼ {cv:.1f}",
                    xy=(xp, lo + span * 0.02),
                    xytext=(xp, lo + span * 0.02),
                    ha="center", va="bottom",
                    fontsize=FONT["annot"], color=STRATEGY_COLORS.get(s, "black"),
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="0.7",
                              linewidth=0.4, alpha=0.9),
                    clip_on=False, zorder=10,
                )
            elif cv > hi:
                ax.annotate(
                    f"▲ {cv:.1f}",
                    xy=(xp, hi - span * 0.02),
                    xytext=(xp, hi - span * 0.02),
                    ha="center", va="top",
                    fontsize=FONT["annot"], color=STRATEGY_COLORS.get(s, "black"),
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="0.7",
                              linewidth=0.4, alpha=0.9),
                    clip_on=False, zorder=10,
                )

        # Caption inside the axes explaining the convention. Tucked into the
        # corner that's least likely to overlap data (upper-left for VAF since
        # the best values sit on the right when sorted descending).
        ax.text(
            0.01, 0.97,
            "▼ off-scale (clipped — see CSV for raw value)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT["annot"], color="0.30", style="italic",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="0.7",
                      linewidth=0.4, alpha=0.9),
        )

    fig.tight_layout()
    _save_png_svg(save_dir / file_stem)
    plt.close(fig)


def plot_trajectory_overlay(
        processed_segments: List[np.ndarray],
        recon_by_strategy: Dict[str, List[np.ndarray]],
        motion_ids: Sequence[int],
        motion_id_to_name: Dict[int, str],
        joint_names: Sequence[str],
        motions_to_show: Sequence[str],
        joints_to_show: Sequence[str],
        save_dir: Path,
) -> None:
    """
    For each motion in ``motions_to_show``, render a single figure with one
    column per shown joint and one row per axis (x/y/z). Each subplot
    overlays the real trajectory (solid black) on top of each strategy's
    reconstruction (colored), so a reader can compare them at a glance.

    Picks the first segment of each motion as the representative example.
    """
    # Map motion_name → first segment idx
    name_to_first_idx: Dict[str, int] = {}
    for i, mid in enumerate(motion_ids):
        nm = motion_id_to_name.get(int(mid), str(mid))
        if nm not in name_to_first_idx:
            name_to_first_idx[nm] = i

    joint_to_idx = {j: k for k, j in enumerate(joint_names)}
    axis_labels = ["x", "y", "z"]

    for motion_name in motions_to_show:
        if motion_name not in name_to_first_idx:
            print(f"[trajectory_overlay] motion '{motion_name}' not in data — skipping")
            continue
        seg_idx = name_to_first_idx[motion_name]
        real = processed_segments[seg_idx]   # [n_signals, T]
        T = real.shape[1]
        time = np.arange(T)

        n_joints_show = len(joints_to_show)
        fig, axes = plt.subplots(
            3, n_joints_show,
            figsize=(3.4 * n_joints_show, 6.8),
            sharex=True,
        )
        if n_joints_show == 1:
            axes = axes.reshape(3, 1)

        for col, j_name in enumerate(joints_to_show):
            if j_name not in joint_to_idx:
                continue
            j_idx = joint_to_idx[j_name]
            for axis_idx in range(3):
                ax = axes[axis_idx, col]
                ch = j_idx * 3 + axis_idx
                ax.plot(time, real[ch], color="black", lw=1.3,
                        label="Real", zorder=5)
                for strat, recons in recon_by_strategy.items():
                    rec = recons[seg_idx]
                    ax.plot(time, rec[ch], color=STRATEGY_COLORS.get(strat, "gray"),
                            lw=1.0, alpha=0.85, label=strat)
                if col == 0:
                    ax.set_ylabel(f"{axis_labels[axis_idx]}",
                                  fontsize=FONT["label"], fontweight="bold")
                if axis_idx == 0:
                    ax.set_title(j_name, fontsize=FONT["title"],
                                 fontweight="bold")
                if axis_idx == 2:
                    ax.set_xlabel("Frame", fontsize=FONT["label"])
                ax.tick_params(axis="both", labelsize=FONT["tick"])
                ax.grid(True, alpha=0.3, linewidth=0.4)

        # Single legend for the whole figure
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=len(handles),
                   frameon=False, fontsize=FONT["legend"])
        fig.suptitle(f"Reconstruction vs. real — {motion_name}",
                     fontsize=FONT["suptitle"], fontweight="bold", y=1.05)
        fig.tight_layout()
        _save_png_svg(save_dir / f"trajectory_overlay_{motion_name}")
        plt.close(fig)


# =============================================================================
# Loaders for the three strategies (defaults match this repo's layout)
# =============================================================================

DEFAULT_DATA_DIR = "../../data/pymotion_position_csv_files"
DEFAULT_MOTION_MAPPING = "../../data/common_motion_mapping.json"
DEFAULT_NUM_MPS = 5
DEFAULT_TPOINTS = 35
DEFAULT_LEGENDRE_MAX_DEGREE = 0
DEFAULT_AE_LATENT_DIM = 32
DEFAULT_AE_HIDDEN_DIM = 128


def _load_motion_id_to_name(mapping_path: str) -> Dict[int, str]:
    try:
        with open(mapping_path, "r") as f:
            data = json.load(f)
        raw = data.get("mapping", data)
        return {int(v): str(k) for k, v in raw.items()}
    except Exception as e:
        print(f"[warning] could not load motion mapping: {e}")
        return {}


def _load_tmp_model(model_dir: str, num_mps: int, tpoints: int,
                    num_segments: int, num_signals: int):
    model_path = os.path.join(model_dir, f"mp_model_{num_mps}_PC_tpoints_{tpoints}")
    return load_model_with_full_state(
        model_path, num_segments=num_segments, num_signals=num_signals,
    )


def _load_ae_model(model_path: str, input_dim: int, max_length: int):
    """Lazy import so users without torch/autoencoder_extraction can still run TMP/Legendre."""
    from mp_movement_classifier.benchmark_analysis.autoencoder_extraction import (
        TemporalAutoencoder,
    )
    model = TemporalAutoencoder(
        input_dim=input_dim,
        hidden_dim=DEFAULT_AE_HIDDEN_DIM,
        latent_dim=DEFAULT_AE_LATENT_DIM,
        max_length=max_length,
        use_lstm=False,
    )
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# =============================================================================
# Main orchestration
# =============================================================================

def run_reconstruction_analysis(
        data_dir: str = DEFAULT_DATA_DIR,
        tmp_model_dir: Optional[str] = None,
        ae_model_path: Optional[str] = None,
        legendre_max_degree: int = DEFAULT_LEGENDRE_MAX_DEGREE,
        num_mps: int = DEFAULT_NUM_MPS,
        tpoints: int = DEFAULT_TPOINTS,
        motion_mapping_path: str = DEFAULT_MOTION_MAPPING,
        save_dir: Optional[str] = None,
        joint_names: Optional[Sequence[str]] = None,
        motions_to_show: Sequence[str] = ("walking", "jumping_jacks", "hand_clapping"),
        joints_to_show: Sequence[str] = ("RKnee", "LWrist", "RAnkle"),
) -> Dict[str, pd.DataFrame]:
    """
    Returns
    -------
    dict with keys ``per_segment``, ``summary``, ``per_class`` —
    the three DataFrames also saved to disk.
    """
    if joint_names is None:
        joint_names = [
            "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
            "Spine", "Thorax", "Neck",
            "LShoulder", "LElbow", "LWrist",
            "RShoulder", "RElbow", "RWrist",
        ]
    n_joints = len(joint_names)

    if save_dir is None:
        save_dir = os.path.join(
            "./../../results/tmp_configs",
            f"new_seg_mp_model_{num_mps}_phase_three",
            "reconstruction_analysis",
        )
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data once (shared by all strategies) ------------------
    print("[1/4] Loading motion data ...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=data_dir, data_type="position", filtering=False,
    )
    motion_id_to_name = _load_motion_id_to_name(motion_mapping_path)
    num_segments = len(processed_segments)
    num_signals = processed_segments[0].shape[0]
    print(f"  segments={num_segments}  signals={num_signals}")

    # ---- 2. Run reconstruction per strategy ---------------------------
    print("[2/4] Running per-strategy reconstruction ...")
    recon_by_strategy: Dict[str, List[np.ndarray]] = {}
    print("  · TMP ...")
    tmp_model = _load_tmp_model(tmp_model_dir, num_mps, tpoints,
                                num_segments, num_signals)
    recon_by_strategy["TMP"] = reconstruct_tmp(tmp_model, processed_segments)

    print(f"  · Legendre (max_degree={legendre_max_degree}) ...")
    recon_by_strategy["Legendre"] = reconstruct_legendre(
        processed_segments, legendre_max_degree,
    )

    if ae_model_path:
        print("  · Autoencoder ...")
        max_length = max(seg.shape[1] for seg in processed_segments)
        ae_model = _load_ae_model(ae_model_path, input_dim=num_signals,
                                  max_length=max_length)
        recon_by_strategy["Autoencoder"] = reconstruct_autoencoder(
            ae_model, processed_segments, device="cpu", max_length=max_length,
        )

    # ---- 3. Compute metrics & aggregate -------------------------------
    print("[3/4] Computing metrics ...")
    frames = []
    for strat, recons in recon_by_strategy.items():
        df = compute_per_segment_metrics(
            strategy_name=strat,
            real_segments=processed_segments,
            recon_segments=recons,
            motion_ids=segment_motion_ids,
            motion_id_to_name=motion_id_to_name,
            n_joints=n_joints,
        )
        frames.append(df)
    per_segment = pd.concat(frames, ignore_index=True)
    summary = summarize_per_strategy(per_segment)
    per_class = summarize_per_class(per_segment)

    per_segment.to_csv(save_dir / "reconstruction_metrics_per_segment.csv", index=False)
    summary.to_csv(save_dir / "reconstruction_metrics_summary.csv", index=False)
    per_class.to_csv(save_dir / "reconstruction_metrics_per_class.csv", index=False)
    print(f"  ✓ Wrote CSVs to {save_dir}")
    print("\n--- Overall summary ---")
    print(summary.to_string(index=False))

    # ---- 4. Figures ----------------------------------------------------
    print("[4/4] Rendering figures ...")
    plot_summary_bars(summary, save_dir)

    # --- Mean ± std versions (clipped y-axis on VAF so a few catastrophic
    # AE outliers don't crush the readable range to one pixel near zero).
    plot_per_class_bars(
        per_class,
        metric_center="vaf_mean", metric_low="vaf_std", metric_high=None,
        ylabel="VAF (higher better)",
        title="Per-class VAF — mean ± std",
        save_dir=save_dir, file_stem="vaf_per_class_mean",
        stat_kind="mean",
        ylim=(-0.5, 1.05),  # clip — see CSV for raw catastrophic values
    )
    plot_per_class_bars(
        per_class,
        metric_center="mpjpe_mean", metric_low="mpjpe_std", metric_high=None,
        ylabel="MPJPE (lower better, input units)",
        title="Per-class MPJPE — mean ± std",
        save_dir=save_dir, file_stem="mpjpe_per_class_mean",
        stat_kind="mean",
    )
    plot_per_class_bars(
        per_class,
        metric_center="vrmse_mean", metric_low="vrmse_std", metric_high=None,
        ylabel="Velocity-RMSE (lower better)",
        title="Per-class velocity-RMSE — mean ± std",
        save_dir=save_dir, file_stem="vrmse_per_class_mean",
        stat_kind="mean",
    )

    # --- Median + IQR versions (robust to heavy-tailed per-segment
    # distributions; recommended for the manuscript's headline figure).
    plot_per_class_bars(
        per_class,
        metric_center="vaf_median", metric_low="vaf_q25", metric_high="vaf_q75",
        ylabel="VAF (higher better)",
        title="Per-class VAF — median + IQR",
        save_dir=save_dir, file_stem="vaf_per_class_median",
        stat_kind="median",
    )
    plot_per_class_bars(
        per_class,
        metric_center="mpjpe_median", metric_low="mpjpe_q25", metric_high="mpjpe_q75",
        ylabel="MPJPE (lower better, input units)",
        title="Per-class MPJPE — median + IQR",
        save_dir=save_dir, file_stem="mpjpe_per_class_median",
        stat_kind="median",
    )
    plot_per_class_bars(
        per_class,
        metric_center="vrmse_median", metric_low="vrmse_q25", metric_high="vrmse_q75",
        ylabel="Velocity-RMSE (lower better)",
        title="Per-class velocity-RMSE — median + IQR",
        save_dir=save_dir, file_stem="vrmse_per_class_median",
        stat_kind="median",
    )
    plot_trajectory_overlay(
        processed_segments=processed_segments,
        recon_by_strategy=recon_by_strategy,
        motion_ids=segment_motion_ids,
        motion_id_to_name=motion_id_to_name,
        joint_names=joint_names,
        motions_to_show=motions_to_show,
        joints_to_show=joints_to_show,
        save_dir=save_dir,
    )

    print(f"\n✓ Done. Artifacts in: {save_dir}")
    return {"per_segment": per_segment, "summary": summary, "per_class": per_class}


def main() -> None:
    # Default paths match the rest of this repo. Override as needed.
    num_mps = DEFAULT_NUM_MPS
    tmp_model_dir = os.path.join(
        "./../../results/tmp_configs",
        f"new_seg_mp_model_{num_mps}_phase_three",
    )
    ae_model_path = os.path.join(
        f"./../../results/tmp_configs/new_seg_mp_model_{num_mps}_phase_three/autoencoder_analysis",
        "models", "best_autoencoder.pt",
    )
    ae_model_path = ae_model_path if os.path.exists(ae_model_path) else None
    if ae_model_path is None:
        print("[info] AE checkpoint not found at default path; skipping AE. "
              "Pass --ae-model-path or run_reconstruction_analysis(ae_model_path=...).")

    run_reconstruction_analysis(
        tmp_model_dir=tmp_model_dir,
        ae_model_path=ae_model_path,
        legendre_max_degree=DEFAULT_LEGENDRE_MAX_DEGREE,
        num_mps=num_mps,
        tpoints=DEFAULT_TPOINTS,
    )


if __name__ == "__main__":
    main()
