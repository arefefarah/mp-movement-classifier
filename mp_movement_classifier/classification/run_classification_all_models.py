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
DPI_EXPORT = 400

from mp_movement_classifier.utils.utils import (
    process_motion_data,
    load_model_with_full_state,
)

from mp_movement_classifier.classification.utils import (
    prepare_weights_for_classification,
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


def _plot_combined_pca_variance(explained_by_model: dict, out_dir: Path, upto: int = 80) -> Path:
    """
    Plot cumulative explained variance curves for multiple models on one figure.
    explained_by_model: dict mapping model name -> 1D np.ndarray of explained_variance_ratio_.
    """
    if not explained_by_model:
        raise ValueError("No PCA explained variance data provided for combined plot")

    # Determine the maximum number of components to consider across all models
    max_len = min(upto, max(len(v) for v in explained_by_model.values()))
    if max_len <= 0:
        raise ValueError("Explained variance arrays are empty")

    plt.figure(figsize=FIG_SIZE_COMBINED)  # ~2250x1860 px at 300 dpi

    colors = {
        'TMP': '#1f77b4',
        'AE': '#ff7f0e',
        'Legendre': '#2ca02c',
    }

    # Plot each model with its own x-range to avoid dimension mismatch
    for name, ratios in explained_by_model.items():
        if ratios is None or len(ratios) == 0:
            continue
        upto_i = min(max_len, len(ratios))
        x_i = np.arange(1, upto_i + 1)
        cumsum_i = np.cumsum(ratios[:upto_i])
        plt.plot(x_i, cumsum_i, label=name, linewidth=3.0, marker='o', markersize=6, color=colors.get(name))

    plt.axhline(y=0.90, color='r', linestyle='--', linewidth=2.0, label='90%')
    plt.axhline(y=0.95, color='g', linestyle='--', linewidth=2.0, label='95%')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative PCA Explained Variance (Combined)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _ensure_dir(out_dir)
    out_path = out_dir / 'pca_cumulative_variance_combined.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI_EXPORT, bbox_inches='tight', facecolor='white')
    plt.close()
    return out_path


def _plot_combined_pca_histograms(explained_by_model: dict, out_dir: Path, upto: int = 20) -> Path:
    """
    Create a single figure (no subplots) that contains THREE histograms together:
    per-component explained variance ratios for TMP, AE, and Legendre on one Axes.
    Bars are grouped side-by-side per component index using small x-offsets.
    Colors are consistent with the combined variance plot. No 90%/95% lines.
    """
    if not explained_by_model:
        raise ValueError("No PCA explained variance data provided for combined histogram plot")

    # Prepare colors and plotting order
    colors = {
        'TMP': '#1f77b4',
        'AE': '#ff7f0e',
        'Legendre': '#2ca02c',
    }
    model_order = [m for m in ['TMP', 'AE', 'Legendre'] if m in explained_by_model]
    if not model_order:
        model_order = list(explained_by_model.keys())

    # Determine per-model component caps and global axes limits
    upto_per_model = {}
    global_max_x = 0
    global_max_y = 0.0
    for name in model_order:
        ratios = explained_by_model.get(name)
        if ratios is None or len(ratios) == 0:
            upto_per_model[name] = 0
            continue
        upto_i = min(upto, len(ratios))
        upto_per_model[name] = upto_i
        global_max_x = max(global_max_x, upto_i)
        if upto_i > 0:
            global_max_y = max(global_max_y, float(np.max(ratios[:upto_i])))

    # Setup single figure and axes (publication settings handled globally)
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_COMBINED)

    # Compute bar placement: side-by-side groups per PC
    n_series = sum(1 for name in model_order if upto_per_model.get(name, 0) > 0)
    if n_series == 0:
        raise ValueError("No non-empty PCA explained variance arrays provided")

    # Total group width in data units; distribute among series (wider groups)
    group_width = 0.96
    bar_width = group_width / n_series

    # Offsets centered around each integer x position (1..global_max_x)
    # Example for 3 series: offsets = [-bar_width, 0, +bar_width]
    offsets = []
    start = - (n_series - 1) / 2.0 * bar_width
    for i in range(n_series):
        offsets.append(start + i * bar_width)

    # Map model -> offset index for deterministic ordering
    visible_models = [name for name in model_order if upto_per_model.get(name, 0) > 0]

    # Optional: nicer y tick formatting
    try:
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    except Exception:
        pass

    # Plot each model's bars (thicker, higher alpha)
    for idx, name in enumerate(visible_models):
        ratios = explained_by_model[name]
        upto_i = upto_per_model[name]
        if upto_i <= 0:
            continue
        x = np.arange(1, global_max_x + 1)
        # Build y with zeros beyond available PCs so groups stay aligned
        y = np.zeros_like(x, dtype=float)
        y[:upto_i] = ratios[:upto_i]
        ax.bar(x + offsets[idx], y, width=bar_width, color=colors.get(name),
               alpha=0.95, edgecolor='black', linewidth=0.8, label=name)

    # Axes styling
    ax.set_xlim(0.5, global_max_x + 0.5)
    ax.set_ylim(0.0, min(1.0, max(0.05, global_max_y * 1.1)))
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance')
    ax.set_title('Variance Explained by Each PC (Combined)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    _ensure_dir(out_dir)
    out_path = out_dir / 'pca_variance_explained_hist_combined.png'
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI_EXPORT, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def _run_tmp(data_dir: str, tmp_model_dir: str, seed: int,
             primary_classifier: str, also_run_rf: bool):
    print("[TMP] Loading data and model...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=data_dir, data_type='position', filtering=False
    )
    num_segments = len(processed_segments)
    num_signals = processed_segments[0].shape[0]

    model_path = os.path.join(tmp_model_dir, 'mp_model_5_PC_tpoints_30')
    # If a different filename pattern is used, allow the user to provide the full path in tmp_model_dir
    if not os.path.exists(model_path):
        # Try to find a single file starting with 'mp_model_' in the dir
        candidates = [f for f in os.listdir(tmp_model_dir) if f.startswith('mp_model_')]
        if not candidates:
            raise FileNotFoundError(f"No TMP model file found under {tmp_model_dir}. Provide --tmp-model-dir pointing to the trained model directory.")
        model_path = os.path.join(tmp_model_dir, candidates[0])

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
        also_run_random_forest=also_run_rf,
        seed=seed,
        cv_folds=5, perform_cv=True
    )
    print(f"[TMP] Done. Artifacts: {out_dir}")
    # Return the full results dictionary
    return results

def _ae_default_out_dir(ae_model_path: str) -> Path:
    # Place results under the model folder's parent results dir if possible
    p = Path(ae_model_path).resolve()
    return p.parent.parent

def _run_ae(data_dir: str, ae_model_path: str, ae_out_dir: str | None, seed: int,
            primary_classifier: str, also_run_rf: bool, cache_dir: str):
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
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

        train_repr, train_labels = extract_representations(model, train_loader, CONFIG['device'])
        test_repr, test_labels = extract_representations(model, test_loader, CONFIG['device'])

        X_latent = np.vstack([train_repr, test_repr])
        y_latent = np.concatenate([train_labels, test_labels])

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
        also_run_random_forest=also_run_rf,
        seed=seed,
        cv_folds=5, perform_cv=True
    )
    print(f"[AE] Done. Artifacts: {cls_out_dir}")
    # Return the full results dictionary
    return results


def _run_legendre(data_dir: str, legendre_out_dir: str | None, seed: int,
                  primary_classifier: str, also_run_rf: bool):
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
        also_run_random_forest=also_run_rf,
        seed=seed,
        cv_folds=5, perform_cv=True
    )
    print(f"[Legendre] Done. Artifacts: {cls_out_dir}")
    # Return the full results dictionary
    return results


def _plot_cross_validation_comparison(results_by_model: dict, combined_out_dir: Path) -> Path:
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
        if 'cross_validation' in results and 'linear_svc' in results['cross_validation']:
            cv_results = results['cross_validation']['linear_svc']
            cv_data[model_key] = cv_results
        else:
            print(f"Warning: No cross-validation results found for {model_key}")

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

    # Overall styling
    # fig.suptitle('Cross-Validation Performance Comparison\n5-fold CV with LinearSVC',
    #              fontsize=15, fontweight='bold', y=0.96)

    # Adjust subplot spacing for better compact look
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95,
                        wspace=0.25, hspace=0.3)  # Use subplots_adjust instead

    # Save
    out_path = combined_out_dir / 'cross_validation_comparison_focused.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nFigure saved to: {out_path}")


    return out_path


def _create_cv_summary_table(results_by_model: dict, combined_out_dir: Path) -> Path:
    """
    Create a summary table of cross-validation results for easy comparison.
    """
    import pandas as pd

    model_names = {'tmp': 'TMP', 'ae': 'Autoencoder', 'legendre': 'Legendre'}
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    summary_data = []

    for model_key, results in results_by_model.items():
        if 'cross_validation' in results and 'linear_svc' in results['cross_validation']:
            cv_results = results['cross_validation']['linear_svc']
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


def create_cross_validation_comparison(results_by_model: dict, combined_out_dir: Path):

    print("\n" + "=" * 60)
    print("CREATING CROSS-VALIDATION COMPARISON")
    print("=" * 60)

    # Create the comparison plot
    plot_path = _plot_cross_validation_comparison(results_by_model, combined_out_dir)

    # Create summary table
    table_path = _create_cv_summary_table(results_by_model, combined_out_dir)

    if plot_path:
        print(f"Cross-validation comparison plot created: {plot_path}")
    if table_path:
        print(f"Cross-validation summary table created: {table_path}")

    return plot_path, table_path


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
    parser.add_argument('--primary-classifier', type=str, choices=['linear_svc', 'random_forest'], default='linear_svc')
    parser.add_argument('--rf', type=int, choices=[0, 1], default=1, help='Enable (1) or disable (0) secondary RandomForest run.')
    parser.add_argument('--seed', type=int, default=42)

    # Cache location
    parser.add_argument('--cache-dir', type=str, default=DEFAULT_CACHE_DIR, help='Directory to store AE latents cache (defaults to project data/).')

    # Combined plot output directory (optional)
    parser.add_argument('--combined-out-dir', type=str, default=None,
                        help='Directory to save the combined PCA variance plot. '
                             'Defaults to --tmp-model-dir if provided; otherwise results/tmp_configs/new_seg_pymotion_position_mp_model')

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    also_run_rf = bool(args.rf)

    # Store both PCA data for combined plots AND full results for CV comparison
    explained_by_model = {}
    results_by_model = {}

    if 'tmp' in args.models:
        if not args.tmp_model_dir:
            raise ValueError("--tmp-model-dir is required when including 'tmp' in --models")
        tmp_results = _run_tmp(
            data_dir=args.data_dir,
            tmp_model_dir=args.tmp_model_dir,
            seed=args.seed,
            primary_classifier=args.primary_classifier,
            also_run_rf=also_run_rf,
        )
        # Store full results for CV comparison
        results_by_model['tmp'] = tmp_results
        # Extract PCA data for combined plots
        if tmp_results and 'pca' in tmp_results:
            explained_by_model['TMP'] = tmp_results['pca'].get('explained_variance_ratio', None)

    if 'ae' in args.models:
        if not args.ae_model_path:
            raise ValueError("--ae-model-path is required when including 'ae' in --models")
        ae_results = _run_ae(
            data_dir=args.data_dir,
            ae_model_path=args.ae_model_path,
            ae_out_dir=args.ae_out_dir,
            seed=args.seed,
            primary_classifier=args.primary_classifier,
            also_run_rf=also_run_rf,
            cache_dir=args.cache_dir,
        )
        # Store full results for CV comparison
        results_by_model['ae'] = ae_results
        # Extract PCA data for combined plots
        if ae_results and 'pca' in ae_results:
            explained_by_model['AE'] = ae_results['pca'].get('explained_variance_ratio', None)

    if 'legendre' in args.models:
        leg_results = _run_legendre(
            data_dir=args.data_dir,
            legendre_out_dir=args.legendre_out_dir,
            seed=args.seed,
            primary_classifier=args.primary_classifier,
            also_run_rf=also_run_rf,
        )
        # Store full results for CV comparison
        results_by_model['legendre'] = leg_results
        # Extract PCA data for combined plots
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
        out_path_var = _plot_combined_pca_variance(explained_by_model, combined_dir)
        print(f"[Combined] PCA cumulative variance plot saved to: {out_path_var}")
        out_path_hist = _plot_combined_pca_histograms(explained_by_model, combined_dir)
        print(f"[Combined] PCA variance explained histogram (stacked) saved to: {out_path_hist}")
    else:
        print("[Combined] Skipped PCA plots: no PCA outputs available from selected models.")

    # Create cross-validation comparison
    if results_by_model:
        create_cross_validation_comparison(results_by_model, combined_dir)
    else:
        print("[Combined] Skipped CV comparison: no results available from selected models.")


if __name__ == '__main__':
    main()