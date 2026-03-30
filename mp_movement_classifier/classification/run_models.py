from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mp_movement_classifier.utils.utils import (
    process_motion_data,
    load_model_with_full_state,
)

from mp_movement_classifier.classification.classification_utils import (
    prepare_weights_for_classification,
)
from mp_movement_classifier.classification.pipeline import run_classification_pipeline

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

    plt.figure(figsize=(10, 7))

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
        plt.plot(x_i, cumsum_i, label=name, linewidth=2, marker='o', markersize=4, color=colors.get(name))

    plt.axhline(y=0.90, color='r', linestyle='--', linewidth=1.5, label='90%')
    plt.axhline(y=0.95, color='g', linestyle='--', linewidth=1.5, label='95%')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative PCA Explained Variance (Combined)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _ensure_dir(out_dir)
    out_path = out_dir / 'pca_cumulative_variance_combined.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
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
    res = run_classification_pipeline(
        X=X, y=y, out_dir=out_dir,
        feature_names=feature_names,
        feature_structure={'n_signals': num_signals, 'n_features_per_signal': model.num_MPs},
        primary_classifier=primary_classifier,
        also_run_random_forest=also_run_rf,
        seed=seed,
    )
    print(f"[TMP] Done. Artifacts: {out_dir}")
    # Return PCA explained variance ratio for combined plotting
    try:
        return res.get('pca', {}).get('explained_variance_ratio', None)
    except Exception:
        return None


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
    res = run_classification_pipeline(
        X=X_latent, y=y_latent, out_dir=cls_out_dir,
        feature_names=feature_names,
        feature_structure={'n_features': X_latent.shape[1]},
        primary_classifier=primary_classifier,
        also_run_random_forest=also_run_rf,
        seed=seed,
    )
    print(f"[AE] Done. Artifacts: {cls_out_dir}")
    try:
        return res.get('pca', {}).get('explained_variance_ratio', None)
    except Exception:
        return None


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
    res = run_classification_pipeline(
        X=X, y=y, out_dir=cls_out_dir,
        feature_names=feature_names,
        feature_structure={'n_signals': n_signals, 'n_features_per_signal': max_degree + 1},
        primary_classifier=primary_classifier,
        also_run_random_forest=also_run_rf,
        seed=seed,
    )
    print(f"[Legendre] Done. Artifacts: {cls_out_dir}")
    try:
        return res.get('pca', {}).get('explained_variance_ratio', None)
    except Exception:
        return None


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

    explained_by_model = {}

    if 'tmp' in args.models:
        if not args.tmp_model_dir:
            raise ValueError("--tmp-model-dir is required when including 'tmp' in --models")
        tmp_ev = _run_tmp(
            data_dir=args.data_dir,
            tmp_model_dir=args.tmp_model_dir,
            seed=args.seed,
            primary_classifier=args.primary_classifier,
            also_run_rf=also_run_rf,
        )
        if tmp_ev is not None:
            explained_by_model['TMP'] = tmp_ev

    if 'ae' in args.models:
        if not args.ae_model_path:
            raise ValueError("--ae-model-path is required when including 'ae' in --models")
        ae_ev = _run_ae(
            data_dir=args.data_dir,
            ae_model_path=args.ae_model_path,
            ae_out_dir=args.ae_out_dir,
            seed=args.seed,
            primary_classifier=args.primary_classifier,
            also_run_rf=also_run_rf,
            cache_dir=args.cache_dir,
        )
        if ae_ev is not None:
            explained_by_model['AE'] = ae_ev

    if 'legendre' in args.models:
        leg_ev = _run_legendre(
            data_dir=args.data_dir,
            legendre_out_dir=args.legendre_out_dir,
            seed=args.seed,
            primary_classifier=args.primary_classifier,
            also_run_rf=also_run_rf,
        )
        if leg_ev is not None:
            explained_by_model['Legendre'] = leg_ev

    # Determine combined output directory
    if args.combined_out_dir:
        combined_dir = Path(args.combined_out_dir)
    elif args.tmp_model_dir:
        combined_dir = Path(args.tmp_model_dir)
    else:
        combined_dir = Path(__file__).resolve().parents[2] / 'results' / 'tmp_configs' / 'new_seg_pymotion_position_mp_model'

    if explained_by_model:
        out_path = _plot_combined_pca_variance(explained_by_model, combined_dir)
        print(f"[Combined] PCA cumulative variance plot saved to: {out_path}")
    else:
        print("[Combined] Skipped: no PCA outputs available from selected models.")


if __name__ == '__main__':
    main()
