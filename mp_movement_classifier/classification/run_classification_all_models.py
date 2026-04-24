from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication-style defaults: fonts and sizes
matplotlib.rcParams.update({
    'font.family': 'Arial',  # fallback handled by matplotlib if not available
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# Shared export settings to guarantee identical figure dimensions across combined plots
FIG_SIZE_COMBINED = (7.5, 6.2)  # inches

from mp_movement_classifier.utils.utils import (
    process_motion_data,
    load_model_with_full_state,
)

from mp_movement_classifier.classification.utils import (
    prepare_weights_for_classification,
    compute_classification_aic,
)
from mp_movement_classifier.classification.classification_pipeline import run_classification_pipeline

# Optional imports from existing modules to reuse AE/Legendre utilities
from mp_movement_classifier.benchmark_analysis.autoencoder_extraction import (
    MotionDataset,
    prepare_datasets,
    extract_representations,
    TemporalAutoencoder,
)
from mp_movement_classifier.benchmark_analysis.legendre_extraction import (
    process_with_legendre_basis,
    prepare_coefficient_data,
)


DEFAULT_DATA_DIR = str(Path(__file__).resolve().parents[2] / 'data' / 'pymotion_position_csv_files')
DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parents[2] / 'data')


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
def _plot_combined_pca_figure(explained_by_model: dict,
                               out_dir: Path,
                               upto_bars: int = 10,
                               upto_cum: int | None = 80) -> Path:

    from matplotlib.ticker import FormatStrFormatter, MaxNLocator


    colors = {'TMP': '#1f77b4', 'AE': '#ff7f0e', 'Legendre': '#2ca02c'}
    model_order = [m for m in ['TMP', 'AE', 'Legendre'] if m in explained_by_model]
    if not model_order:
        model_order = list(explained_by_model.keys())

    visible = [m for m in model_order
               if explained_by_model.get(m) is not None and len(explained_by_model[m]) > 0]
    if not visible:
        raise ValueError("No non-empty PCA explained variance arrays provided")

    def _pc_at(r, thr):
        c = np.cumsum(np.asarray(r, dtype=float))
        idx = np.where(c >= thr)[0]
        return int(idx[0] + 1) if len(idx) else None

    # Local rcParams — compact fonts suited to double-column manuscript width.
    rc_local = {
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    }
    with plt.rc_context(rc_local):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(7.2, 2.6),
            gridspec_kw={'width_ratios': [0.85, 1.15], 'wspace': 0.28}
        )

        # ---------- (a) per-component bars ----------
        n = len(visible)
        gw, bw = 0.9, 0.9 / n
        offs = np.linspace(-gw/2 + bw/2, gw/2 - bw/2, n)
        x = np.arange(1, upto_bars + 1)
        for i, name in enumerate(visible):
            r = np.asarray(explained_by_model[name], dtype=float)
            y = np.zeros(upto_bars)
            k = min(upto_bars, len(r))
            y[:k] = r[:k]
            ax1.bar(x + offs[i], y, width=bw, color=colors.get(name),
                    edgecolor='black', linewidth=0.3, label=name)

        ax1.set_xlabel('Principal component')
        ax1.set_ylabel('Explained variance')
        ax1.set_xlim(0.4, upto_bars + 0.6)
        ax1.set_xticks(x)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.grid(True, axis='y', alpha=0.3, linewidth=0.4)
        ax1.set_axisbelow(True)
        ax1.legend(frameon=False, loc='upper right', handlelength=1.2)
        ax1.text(-0.14, 1.02, 'a', transform=ax1.transAxes,
                 fontsize=10, fontweight='bold', va='bottom')

        # ---------- (b) cumulative, capped at upto_cum ----------
        actual_max = max(len(explained_by_model[n]) for n in visible)
        max_len = actual_max if upto_cum is None else min(upto_cum, actual_max)

        crossings = []
        for name in visible:
            r = np.asarray(explained_by_model[name], dtype=float)
            k_cap = min(max_len, len(r))
            cum = np.cumsum(r[:k_cap])
            xs = np.arange(1, k_cap + 1)
            ax2.plot(xs, cum, color=colors.get(name), linewidth=1.2, label=name)
            # Small markers only on the first 10 points — anchors the eye without clutter
            m = min(10, k_cap)
            ax2.plot(xs[:m], cum[:m], 'o', color=colors.get(name), markersize=2.2)
            # 90% crossing (use full ratios, not truncated, in case 90% is past upto_cum)
            k90 = _pc_at(r, 0.90)
            if k90 is not None and k90 <= max_len:
                crossings.append((name, k90))
                ax2.plot([k90, k90], [0, 0.9], color=colors.get(name),
                         linewidth=0.6, linestyle=':', alpha=0.7)
                ax2.plot(k90, 0.9, 'o', color=colors.get(name), markersize=4,
                         markeredgecolor='black', markeredgewidth=0.4, zorder=5)
            elif k90 is not None:
                # 90% threshold is beyond visible range — note it in the legend text
                crossings.append((name, k90))

        # Threshold lines
        for thr, ls, lbl in [(0.90, '--', '90%'), (0.95, ':', '95%')]:
            ax2.axhline(thr, color='gray', linestyle=ls, linewidth=0.6, alpha=0.8)
            ax2.text(max_len * 1.01, thr, lbl, fontsize=6, color='gray', va='center')

        if crossings:
            lines = [r'$n$ for 90% variance:'] + [f'  {n}: {k}' for n, k in crossings]
            ax2.text(0.97, 0.05, '\n'.join(lines),
                     transform=ax2.transAxes, fontsize=7, color='0.15',
                     ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                               edgecolor='0.7', linewidth=0.4, alpha=0.95))

        ax2.set_xlabel('Number of components')
        ax2.set_ylabel('Cumulative explained variance')
        ax2.set_xlim(0.5, max_len + 0.5)
        ax2.set_ylim(0, 1.02)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.grid(True, alpha=0.3, linewidth=0.4)
        ax2.set_axisbelow(True)
        ax2.text(-0.12, 1.02, 'b', transform=ax2.transAxes,
                 fontsize=10, fontweight='bold', va='bottom')

        _ensure_dir(out_dir)
        out_path = out_dir / 'pca_variance_combined.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
        plt.close(fig)

    return out_path


def _run_tmp(data_dir: str, tmp_model_dir: str, seed: int,
             primary_classifier: str):
    print("[TMP] Loading data and model...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=data_dir, data_type='position', filtering=False
    )
    num_segments = len(processed_segments)
    num_signals = processed_segments[0].shape[0]

    model_path = os.path.join(tmp_model_dir, 'mp_model_5_PC_tpoints_30')
    # # If a different filename pattern is used, allow the user to provide the full path in tmp_model_dir
    # if not os.path.exists(model_path):
    #     # Try to find a single file starting with 'mp_model_' in the dir
    #     candidates = [f for f in os.listdir(tmp_model_dir) if f.startswith('mp_model_')]
    #     if not candidates:
    #         raise FileNotFoundError(f"No TMP model file found under {tmp_model_dir}. Provide --tmp-model-dir pointing to the trained model directory.")
    #     model_path = os.path.join(tmp_model_dir, candidates[0])

    model = load_model_with_full_state(
        model_path, num_segments=num_segments, num_signals=num_signals
    )

    print("[TMP] Preparing features and labels...")
    X = prepare_weights_for_classification(model, num_segments, num_signals, num_MPs=model.num_MPs)
    y = np.array(segment_motion_ids)

    # Feature names
    feature_names = [f"signal_{s}_mp_{m}" for s in range(num_signals) for m in range(model.num_MPs)]

    out_dir = Path(tmp_model_dir) / 'classification'
    _ensure_dir(out_dir)

    print("[TMP] Running unified classification pipeline...")
    results = run_classification_pipeline(
        X=X, y=y, out_dir=out_dir,
        feature_names=feature_names,
        feature_structure={'n_signals': num_signals, 'n_features_per_signal': model.num_MPs},
        primary_classifier=primary_classifier,
        run_all_classifiers=True,
        seed=seed,
        cv_folds=5, perform_cv=True
    )
    print(f"[TMP] Done. Artifacts: {out_dir}")
    return results, X, y

def _ae_default_out_dir(ae_model_path: str) -> Path:
    # Place results under the model folder's parent results dir if possible
    p = Path(ae_model_path).resolve()
    return p.parent.parent

def _run_ae(data_dir: str, ae_model_path: str, ae_out_dir: str | None, seed: int,
            primary_classifier: str, cache_dir: str):
    print("[AE] Loading data...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=data_dir, data_type='position', filtering=False
    )
    # Flip to [time, features] as in autoencoder_extraction
    segments_tf = [seg.T for seg in processed_segments]

    print("[AE] Preparing datasets...")
    train_dataset, val_dataset, test_dataset, scaler = prepare_datasets(
        segments_tf, segment_motion_ids, test_size=0.2, val_size=0.1, normalize=False
    )

    actual_input_dim = train_dataset.n_features

    # Build model architecture consistent with autoencoder_extraction defaults
    CONFIG = {
        'input_dim': actual_input_dim,
        'hidden_dim': 128,
        'latent_dim': 32,  # adjust if your checkpoint differs
        'use_lstm': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print("[AE] Initializing model and loading checkpoint...")
    model = TemporalAutoencoder(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        latent_dim=CONFIG['latent_dim'],
        max_length=train_dataset.max_length,
        use_lstm=CONFIG['use_lstm'],
    )

    checkpoint = torch.load(ae_model_path, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(CONFIG['device'])

    # Try cache first
    cache_path = Path(cache_dir) / 'ae_latents_cache.npz'
    X_latent = None
    y_latent = None
    if cache_path.exists():
        try:
            cache = np.load(cache_path)
            X_latent = cache['X']
            y_latent = cache['y']
            print(f"[AE] Loaded cached latents from {cache_path}")
        except Exception as e:
            print(f"[AE] Failed to load cache ({e}), will recompute.")

    if X_latent is None:
        print("[AE] Extracting latent representations (no training)...")
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

        #make same number of inputs for ae as legendre/TMP model
        train_repr, train_labels = extract_representations(model, train_loader, CONFIG['device'])
        val_repr,   val_labels   = extract_representations(model, val_loader,   CONFIG['device'])
        test_repr,  test_labels  = extract_representations(model, test_loader,  CONFIG['device'])

        X_latent = np.vstack([train_repr, val_repr, test_repr])
        y_latent = np.concatenate([train_labels, val_labels, test_labels])

        try:
            np.savez(cache_path, X=X_latent, y=y_latent)
            print(f"[AE] Cached latents written to {cache_path}")
        except Exception as e:
            print(f"[AE] Warning: could not write cache to {cache_path}: {e}")

    feature_names = [f'latent_{i}' for i in range(X_latent.shape[1])]

    out_dir = Path(ae_out_dir) if ae_out_dir else _ae_default_out_dir(ae_model_path)
    cls_out_dir = out_dir / 'classification'
    _ensure_dir(cls_out_dir)

    print("[AE] Running unified classification pipeline...")
    results = run_classification_pipeline(
        X=X_latent, y=y_latent, out_dir=cls_out_dir,
        feature_names=feature_names,
        feature_structure={'n_features': X_latent.shape[1]},
        primary_classifier=primary_classifier,
        run_all_classifiers=True,
        seed=seed,
        cv_folds=5, perform_cv=True
    )
    print(f"[AE] Done. Artifacts: {cls_out_dir}")
    return results, X_latent, y_latent


def _run_legendre(data_dir: str, legendre_out_dir: str | None, seed: int,
                  primary_classifier: str):
    print("[Legendre] Loading data and computing coefficients...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=data_dir, data_type='position', filtering=False
    )

    max_degree = 1  # adjust if needed
    coefficients, errors = process_with_legendre_basis(processed_segments, max_degree)

    X, y = prepare_coefficient_data(coefficients, segment_motion_ids)

    # Build feature names: deg_k_signal_j
    n_signals = coefficients[0].shape[0]
    feature_names = []
    for j in range(n_signals):
        for k in range(max_degree + 1):
            feature_names.append(f"deg_{k}_signal_{j}")

    out_dir = Path(legendre_out_dir) if legendre_out_dir else (Path(__file__).resolve().parents[2] / 'results' / 'legendre_analysis')
    cls_out_dir = out_dir / 'classification'
    _ensure_dir(cls_out_dir)

    print("[Legendre] Running unified classification pipeline...")
    results = run_classification_pipeline(
        X=X, y=y, out_dir=cls_out_dir,
        feature_names=feature_names,
        feature_structure={'n_signals': n_signals, 'n_features_per_signal': max_degree + 1},
        primary_classifier=primary_classifier,
        run_all_classifiers=True,
        seed=seed,
        cv_folds=5, perform_cv=True
    )
    print(f"[Legendre] Done. Artifacts: {cls_out_dir}")
    return results, X, y


def _plot_cross_validation_comparison(results_by_model: dict, combined_out_dir: Path,
                                      classifier_key: str = 'random_forest') -> Path:
    """
    Create a focused, publication-quality comparison plot with optimized y-axis ranges and compact model spacing.
    """
    # Set style
    plt.style.use('default')
    sns.set_palette("Set2")

    # Extract CV results
    cv_data = {}
    model_names = {'tmp': 'TMP', 'ae': 'Autoencoder', 'legendre': 'Legendre'}
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    metric_labels = {'accuracy': 'Accuracy', 'f1_macro': 'F1-Score',
                     'precision_macro': 'Precision', 'recall_macro': 'Recall'}

    for model_key, results in results_by_model.items():
        if 'cross_validation' in results and classifier_key in results['cross_validation']:
            cv_results = results['cross_validation'][classifier_key]
            cv_data[model_key] = cv_results
        else:
            print(f"Warning: No CV results for {model_key} with classifier {classifier_key}")

    if not cv_data:
        print("No cross-validation data available for plotting")
        return None

    # Calculate global min/max for focused y-axis
    all_scores = []
    for model_key, cv_results in cv_data.items():
        for metric in metrics:
            scores_key = f'{metric}_test_scores'
            if scores_key in cv_results:
                all_scores.extend(cv_results[scores_key])

    global_min = min(all_scores) - 0.01
    global_max = max(all_scores) + 0.01
    y_range = global_max - global_min
    y_padding = y_range * 0.1

    # Create figure with same size but more compact internal spacing
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()

    # Modern color palette
    colors = {'tmp': '#3498db', 'ae': '#e74c3c', 'legendre': '#2ecc71'}

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Prepare data
        plot_data = []
        plot_labels = []
        plot_colors = []
        means = []

        for model_key in ['tmp', 'ae', 'legendre']:
            if model_key in cv_data:
                scores_key = f'{metric}_test_scores'
                if scores_key in cv_data[model_key]:
                    scores = cv_data[model_key][scores_key]
                    plot_data.append(scores)
                    plot_labels.append(model_names[model_key])
                    plot_colors.append(colors[model_key])
                    means.append(np.mean(scores))

        if not plot_data:
            continue

        # Create violin plots with COMPACT positioning
        # Reduce spacing by using closer positions and smaller widths
        positions = np.arange(1, len(plot_data) + 1) * 0.8  # Compress positions by 0.8x
        violin_parts = ax.violinplot(plot_data, positions=positions, widths=0.5,  # Narrower violins
                                     showmeans=False, showmedians=True, showextrema=False)

        # Customize violin colors
        for pc, color in zip(violin_parts['bodies'], plot_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)

        # Add mean markers with values - adjust positions to match violins
        for j, (mean_val, color, label) in enumerate(zip(means, plot_colors, plot_labels)):
            pos = positions[j]  # Use the same compressed positions
            ax.scatter(pos, mean_val, color=color, s=100, marker='D',
                       edgecolor='white', linewidth=2, zorder=5)
            ax.text(pos, mean_val + y_range * 0.05, f'{mean_val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Set focused y-axis range
        ax.set_ylim(global_min - y_padding, global_max + y_padding)

        # Adjust x-axis limits to be more compact around the data
        if positions.size > 0:
            x_margin = 0.3 # Reduced margin for more compact appearance
            ax.set_xlim(positions[0] - x_margin, positions[-1] + x_margin)

        # Styling
        ax.set_title(metric_labels[metric], fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add subtle background
        ax.set_facecolor('#fafafa')


    # Adjust subplot spacing for better compact look
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95,
                        wspace=0.25, hspace=0.3)  # Use subplots_adjust instead

    # Save
    out_path = combined_out_dir / f'cross_validation_comparison_{classifier_key}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nFigure saved to: {out_path}")


    return out_path


def _create_cv_summary_table(results_by_model: dict, combined_out_dir: Path,
                             classifier_key: str = 'random_forest') -> Path:
    """
    Create a summary table of cross-validation results for easy comparison.
    """
    import pandas as pd

    model_names = {'tmp': 'TMP', 'ae': 'Autoencoder', 'legendre': 'Legendre'}
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    summary_data = []

    for model_key, results in results_by_model.items():
        if 'cross_validation' in results and classifier_key in results['cross_validation']:
            cv_results = results['cross_validation'][classifier_key]
            row = {'Model': model_names.get(model_key, model_key)}

            for metric in metrics:
                mean_key = f'{metric}_test_mean'
                std_key = f'{metric}_test_std'
                gap_key = f'{metric}_generalization_gap'

                if mean_key in cv_results:
                    mean_val = cv_results[mean_key]
                    std_val = cv_results[std_key]
                    gap_val = cv_results.get(gap_key, 0)

                    row[f'{metric.title()}_Mean'] = f"{mean_val:.4f}"
                    row[f'{metric.title()}_Std'] = f"{std_val:.4f}"
                    row[f'{metric.title()}_Gap'] = f"{gap_val:.4f}"

            summary_data.append(row)

    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = combined_out_dir / 'cross_validation_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"CV summary table saved to: {csv_path}")
        return csv_path

    return None


def create_cross_validation_comparison(results_by_model: dict, combined_out_dir: Path,
                                       classifier_key: str = 'random_forest'):
    print("\n" + "=" * 60)
    print("CREATING CROSS-VALIDATION COMPARISON")
    print("=" * 60)
    plot_path = _plot_cross_validation_comparison(results_by_model, combined_out_dir, classifier_key)
    table_path = _create_cv_summary_table(results_by_model, combined_out_dir, classifier_key)

    if plot_path:
        print(f"Cross-validation comparison plot created: {plot_path}")
    if table_path:
        print(f"Cross-validation summary table created: {table_path}")

    return plot_path, table_path


def run_aic_comparison(
    X_by_model: dict,
    combined_out_dir: Path,
    seed: int = 42,
    cv_folds: int = 5,
) -> tuple:
    """
    AIC-based model comparison following Burnham & Anderson (2004).

    X_by_model: dict mapping model key ('tmp'/'ae'/'legendre') -> (X, y) tuple
    combined_out_dir: same output directory used for the CV comparison

    AIC = 2k - 2*ln(L_cv)
      k        = number of input features (representation dimensionality)
      ln(L_cv) = cross-validated log-likelihood (5-fold, RF predict_proba)

    Saves:
      aic_comparison_results.csv
      aic_comparison_report.txt
    """
    import pandas as pd

    print("\n" + "=" * 60)
    print("COMPUTING AIC-BASED MODEL COMPARISON")
    print("=" * 60)

    model_display = {'tmp': 'TMP', 'ae': 'Autoencoder', 'legendre': 'Legendre'}

    aic_data = {}
    for model_key, (X, y) in X_by_model.items():
        name = model_display.get(model_key, model_key)
        print(f"[AIC] {name}: k={X.shape[1]}, n_samples={X.shape[0]} — running {cv_folds}-fold CV...")
        result = compute_classification_aic(X, y, seed=seed, cv_folds=cv_folds)
        aic_data[model_key] = result
        print(f"        ln(L_cv)={result['log_likelihood']:.4f}  AIC={result['aic']:.4f}")

    min_aic = min(v['aic'] for v in aic_data.values())

    rows = []
    for model_key, data in aic_data.items():
        delta_i = data['aic'] - min_aic
        p_i = float(np.exp(-delta_i / 2.0))

        if delta_i < 2.0:
            evidence = "Substantial support (Δ < 2)"
        elif delta_i < 4.0:
            evidence = "Strong support (2 < Δ < 4)"
        elif delta_i < 7.0:
            evidence = "Considerably less support (4 < Δ < 7)"
        elif delta_i < 10.0:
            evidence = "Some support (7 < Δ < 10)"
        else:
            evidence = "Essentially no support (Δ > 10)"

        rows.append({
            'Model': model_display.get(model_key, model_key),
            'k_features': data['k'],
            'n_samples': data['n_samples'],
            'log_likelihood_cv': round(data['log_likelihood'], 4),
            'AIC': round(data['aic'], 4),
            'delta_AIC': round(delta_i, 4),
            'p_i': round(p_i, 6),
            'evidence_Burnham_Anderson_2004': evidence,
        })

    rows.sort(key=lambda r: r['AIC'])

    _ensure_dir(combined_out_dir)
    csv_path = combined_out_dir / 'aic_comparison_results.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n[AIC] Results CSV saved to: {csv_path}")

    report_path = combined_out_dir / 'aic_comparison_report.txt'
    best_model = rows[0]['Model']
    with open(report_path, 'w') as f:
        f.write("AIC-Based Model Comparison\n")
        f.write("=" * 70 + "\n")
        f.write("Reference : Burnham & Anderson (2004)\n")
        f.write(f"Method    : {cv_folds}-fold cross-validated log-likelihood\n")
        f.write("Classifier: Random Forest (n_estimators=200)\n")
        f.write("k         : number of input features (representation dimensionality)\n")
        f.write("AIC       : 2k - 2*ln(L_cv)\n")
        f.write("Δ_i       : AIC_i - AIC_min  (0 for the best model)\n")
        f.write("p_i       : exp(-Δ_i / 2)  — relative probability that model i minimises AIC\n")
        f.write("\n")
        f.write("Results (sorted by AIC, lower is better):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<14} {'k':>6} {'n_samples':>10} {'ln(L_cv)':>12} "
                f"{'AIC':>12} {'Δ_AIC':>10} {'p_i':>8}\n")
        f.write("-" * 70 + "\n")
        for row in rows:
            f.write(
                f"{row['Model']:<14} "
                f"{row['k_features']:>6} "
                f"{row['n_samples']:>10} "
                f"{row['log_likelihood_cv']:>12.4f} "
                f"{row['AIC']:>12.4f} "
                f"{row['delta_AIC']:>10.4f} "
                f"{row['p_i']:>8.6f}\n"
            )
        f.write("\n")
        f.write("Evidence categories (Burnham & Anderson 2004):\n")
        f.write("-" * 70 + "\n")
        for row in rows:
            f.write(f"  {row['Model']}: {row['evidence_Burnham_Anderson_2004']}\n")
        f.write("\n")
        f.write("Interpretation guide:\n")
        f.write("  Δ_AIC < 2      Substantial support — model highly likely to be a proper description\n")
        f.write("  2 < Δ_AIC < 4  Strong support\n")
        f.write("  4 < Δ_AIC < 7  Considerably less support\n")
        f.write("  Δ_AIC > 10     Essentially no support\n")
        f.write("\n")
        f.write(f"Best model: {best_model}  (Δ_AIC = 0.0, p_i = 1.000000)\n")
        f.write("\n")
        f.write("Note: AIC penalises extra parameters (k), so a simpler representation\n")
        f.write("is preferred unless a more complex one provides a substantially better\n")
        f.write("cross-validated log-likelihood.  p_i gives the probability that model i\n")
        f.write("would minimise AIC in a replicated experiment.\n")

    print(f"[AIC] Report saved to: {report_path}")
    return csv_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified classification on selected models without retraining.")
    parser.add_argument('--models', nargs='+', choices=['tmp', 'ae', 'legendre'], default=['tmp', 'ae', 'legendre'],
                        help='Subset of models to run (default: all).')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Path to the motion CSV data directory.')

    # TMP
    parser.add_argument('--tmp-model-dir', type=str, help='Directory containing the trained TMP model file (mp_model_...).')

    # Autoencoder
    parser.add_argument('--ae-model-path', type=str, help='Path to best_autoencoder.pt checkpoint for AE.')
    parser.add_argument('--ae-out-dir', type=str, default=None, help='Output directory root for AE results (optional).')

    # Legendre
    parser.add_argument('--legendre-out-dir', type=str, default=None, help='Output directory root for Legendre results (optional).')

    # Classification controls
    parser.add_argument('--primary-classifier', type=str, choices=['linear_svc', 'random_forest'], default='random_forest')
    parser.add_argument('--seed', type=int, default=42)

    # Cache location
    parser.add_argument('--cache-dir', type=str, default=DEFAULT_CACHE_DIR, help='Directory to store AE latents cache (defaults to project data/).')

    # AIC comparison
    parser.add_argument('--run-aic', action='store_true', default=False,
                        help='Compute AIC-based model comparison (cross-validated log-likelihood, RF classifier).')

    # Combined plot output directory (optional)
    parser.add_argument('--combined-out-dir', type=str, default=None,
                        help='Directory to save the combined PCA variance plot. '
                             'Defaults to --tmp-model-dir if provided; otherwise results/tmp_configs/new_seg_pymotion_position_mp_model')

    return parser.parse_args()

def main() -> None:
    args = parse_args()


    # Store PCA data for combined plots, full results for CV comparison,
    # and (X, y) pairs needed for AIC comparison.
    explained_by_model = {}
    results_by_model = {}
    X_by_model = {}  # model_key -> (X, y) for AIC computation

    if 'tmp' in args.models:
        if not args.tmp_model_dir:
            raise ValueError("--tmp-model-dir is required when including 'tmp' in --models")
        tmp_results, tmp_X, tmp_y = _run_tmp(
            data_dir=args.data_dir,
            tmp_model_dir=args.tmp_model_dir,
            seed=args.seed,
            primary_classifier=args.primary_classifier,
        )
        results_by_model['tmp'] = tmp_results
        X_by_model['tmp'] = (tmp_X, tmp_y)
        if tmp_results and 'pca' in tmp_results:
            explained_by_model['TMP'] = tmp_results['pca'].get('explained_variance_ratio', None)

    if 'ae' in args.models:
        if not args.ae_model_path:
            raise ValueError("--ae-model-path is required when including 'ae' in --models")
        ae_results, ae_X, ae_y = _run_ae(
            data_dir=args.data_dir,
            ae_model_path=args.ae_model_path,
            ae_out_dir=args.ae_out_dir,
            seed=args.seed,
            primary_classifier=args.primary_classifier,
            cache_dir=args.cache_dir,
        )
        results_by_model['ae'] = ae_results
        X_by_model['ae'] = (ae_X, ae_y)
        if ae_results and 'pca' in ae_results:
            explained_by_model['AE'] = ae_results['pca'].get('explained_variance_ratio', None)

    if 'legendre' in args.models:
        leg_results, leg_X, leg_y = _run_legendre(
            data_dir=args.data_dir,
            legendre_out_dir=args.legendre_out_dir,
            seed=args.seed,
            primary_classifier=args.primary_classifier
        )
        results_by_model['legendre'] = leg_results
        X_by_model['legendre'] = (leg_X, leg_y)
        if leg_results and 'pca' in leg_results:
            explained_by_model['Legendre'] = leg_results['pca'].get('explained_variance_ratio', None)

    # Determine combined output directory
    if args.combined_out_dir:
        combined_dir = Path(args.combined_out_dir)
    elif args.tmp_model_dir:
        combined_dir = Path(args.tmp_model_dir)
    else:
        combined_dir = Path(__file__).resolve().parents[2] / 'results' / 'tmp_configs' / 'new_seg_pymotion_position_mp_model'

    # Create combined PCA plots
    if explained_by_model:
        out_path_var = _plot_combined_pca_figure(explained_by_model, combined_dir)

    else:
        print("[Combined] Skipped PCA plots: no PCA outputs available from selected models.")

    # Create cross-validation comparison
    if results_by_model:
        create_cross_validation_comparison(results_by_model, combined_dir,
                                           classifier_key=args.primary_classifier)
    else:
        print("[Combined] Skipped CV comparison: no results available from selected models.")

    # AIC-based model comparison
    if args.run_aic and X_by_model:
        run_aic_comparison(X_by_model, combined_dir, seed=args.seed, cv_folds=5)
    elif args.run_aic:
        print("[AIC] Skipped: no model outputs available.")


if __name__ == '__main__':
    main()