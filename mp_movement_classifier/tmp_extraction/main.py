import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from TMP_model import MP_model
import torch_hessian
from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_bvh_data,
    process_motion_data,
    read_bvh_files,
    save_model_with_full_state,
    segment_motion_trajectories,
    parse_bvh_robust,
)
from mp_movement_classifier.benchmark_analysis.posture_removal_experiment import subtract_segment_means
from mp_movement_classifier.utils.plotting import (
    plot_learn_curve, plot_mp,
    plot_reconstructions,
    set_figures_directory
)
from mp_movement_classifier.utils import config
from mp_movement_classifier.classification.utils import prepare_weights_for_classification
from mp_movement_classifier.classification.classification_pipeline import run_classification_pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate MP model on BVH data.")
    parser.add_argument("--num-mps", type=int, default=5, help="Number of movement primitives.")
    parser.add_argument("--num-t-points", type=int, default=20, help="Number of time discretization points.")
    parser.add_argument("--adam-steps", type=int, default=100, help="Number of ADAM optimization steps.")
    parser.add_argument("--bfgs-steps", type=int, default=30, help="Number of BFGS optimization steps.")
    parser.add_argument("--pca-init", action="store_true", default=True, help="Use PCA initialization.")
    parser.add_argument("--no-pca-init", dest="pca_init", action="store_false", help="Disable PCA initialization.")
    parser.add_argument("--load", action="store_true", help="Load an existing model if available.")
    parser.add_argument("--optimize", action="store_true", default=False,
                        help="Run optimization sweeps over num_t_points and num_mps.")
    parser.add_argument("--t-points-values", type=int, nargs='+', default=[10, 15, 20, 25, 30, 35, 40],
                        help="num_t_points values to sweep (num_MPs held at --num-mps).")
    parser.add_argument("--num-mps-values", type=int, nargs='+', default=[2, 5, 8, 11, 14, 17, 20],
                        help="num_mps values to sweep (num_t_points held at --num-t-points).")

    return parser.parse_args()


def prepare_save_paths(num_mps: int, num_t_points: int, model_name_suffix: Optional[str]) -> Tuple[str, str, str]:
    """
    Prepare save paths for model and figures.
    """
    model_dir = os.path.join("./../../results/tmp_configs",
                             f"new_seg_mp_model_{num_mps}_phase_three")
    # model_dir = os.path.join("./../../results/tmp_configs", f"new_seg_mp_model_{num_mps}_tpoints_{num_t_points}_phase_three")
    os.makedirs(model_dir, exist_ok=True)

    model_name = f"mp_model_{num_mps}_PC_tpoints_{num_t_points}"
    if model_name_suffix:
        model_name = f"{model_name}_{model_name_suffix}"

    model_path = os.path.join(model_dir, model_name)

    return model_name, model_path, model_dir


def initialize_or_load_model(
        num_t_points: int,
        num_mps: int,
        num_segments: int,
        num_signals: int,
        processed_data: List[np.ndarray],
        use_pca: bool,
        model_path: str,
        load_existing: bool,
) -> MP_model:
    if load_existing:
        try:
            model = load_model_with_full_state(
                model_path,
                num_segments=num_segments,
                num_signals=num_signals,
            )
            print(f"Loaded model from: {model_path}")
            return model
        except Exception as e:
            print(f"Warning: Failed to load model from {model_path}. Reinitializing. Reason: {e}")
    else:
        init_data = processed_data if use_pca else None
        model = MP_model(
            num_t_points=num_t_points,
            num_MPs=num_mps,
            num_segments=num_segments,
            num_signals=num_signals,
            init_data=init_data,
        )
        print("Initialized a new MP_model instance.")
        return model


def train_and_save(
        model: MP_model,
        processed_data: List[np.ndarray],
        model_path: str,
        adam_steps: int,
        bfgs_steps: int,
) -> None:
    print(f"Training model: ADAM steps={adam_steps}, BFGS steps={bfgs_steps}")
    model.learn(processed_data, adam_steps=adam_steps, bfgs_steps=bfgs_steps)
    print("Training complete. Saving model...")
    save_model_with_full_state(model, model_path)
    print(f"Model saved to: {model_path}")


def evaluate_and_plot(
        model: MP_model,
        processed_data: List[np.ndarray],
        model_name: str,
        model_dir: str,
        tail_window: int,
) -> None:
    """
    Evaluate model and create plots in the model-specific figures directory.
    """
    # Set the figures directory to be inside the model directory
    figures_dir = os.path.join(model_dir, 'primitives')
    set_figures_directory(figures_dir)

    print(f"Saving TMP model figures to: {figures_dir}")

    # Plot MPs
    plot_mp(torch.stack(list(model.MPs)), model_name, save=True)

    # Reconstructions for shortest and longest segments
    segment_lengths = np.array([segment.shape[1] for segment in processed_data], dtype=int)
    recon_data = model.predict(segment_lengths, as_numpy=True)

    motion_name = "all"
    if recon_data is not None and len(recon_data) == len(processed_data):
        try:
            max_idx = segment_lengths.argmax()
            min_idx = segment_lengths.argmin()
            plot_reconstructions(
                processed_data[max_idx],
                recon_data[max_idx],
                f"{motion_name} - max seg length={segment_lengths[max_idx]}",
                save=True,
            )
            plot_reconstructions(
                processed_data[min_idx],
                recon_data[min_idx],
                f"{motion_name} - min seg length={segment_lengths[min_idx]}",
                save=True,
            )
            print("✓ Reconstruction plots saved")
        except Exception as e:
            print(f"Warning: Failed to plot reconstructions. Reason: {e}")

    # Learning curves
    lc = model.learn_curve
    vc = model.VAF_curve
    epochs = np.arange(len(lc))
    plot_learn_curve(epochs, lc, vc, f"TMP model with PCA init", save=True)
    if tail_window > 0 and len(epochs) > tail_window:
        plot_learn_curve(
            epochs[-tail_window:],
            lc[-tail_window:],
            vc[-tail_window:],
            f"tail of {tail_window:d}, PCA init",
            save=True,
        )


def _run_sweep(
        processed_data: List[np.ndarray],
        num_mps: int,
        num_t_points: int,
        adam_steps: int,
        bfgs_steps: int,
) -> Tuple[float, float]:
    """Train one TMP configuration; return (lap_score, vaf)."""
    num_segments = len(processed_data)
    num_signals = processed_data[0].shape[0]
    model = MP_model(
        num_t_points=num_t_points,
        num_MPs=num_mps,
        num_segments=num_segments,
        num_signals=num_signals,
        init_data=processed_data,
    )
    model.learn(processed_data, adam_steps=adam_steps, bfgs_steps=bfgs_steps)
    final_vaf = float(model.VAF_curve[-1])
    lap_score = float(model.Laplace_approx(processed_data))
    return lap_score, final_vaf


def _save_sweep_outputs(
        results: dict,
        sweep_var: str,
        fixed_desc: str,
        save_dir: str,
        file_stem: str,
) -> None:
    """Write txt report and 2-subplot figure (LAP + VAF) for a sweep."""
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    xs = results[sweep_var]
    lap = results['lap_scores']
    vaf = results['vaf_scores']
    best_idx = int(np.argmax(lap))
    best_x = xs[best_idx]

    txt_path = os.path.join(save_dir, f"{file_stem}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"TMP sweep over {sweep_var}\n")
        f.write(f"Fixed: {fixed_desc}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{sweep_var:<15} {'LAP Score':<15} {'VAF':<10}\n")
        f.write("-" * 42 + "\n")
        for i, x in enumerate(xs):
            marker = " <- BEST" if i == best_idx else ""
            f.write(f"{x:<15} {lap[i]:<15.2f} {vaf[i]:<10.4f}{marker}\n")
        f.write(f"\nBest {sweep_var}: {best_x}\n")
    print(f"Results saved to: {txt_path}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].plot(xs, lap, 'o-', linewidth=2, markersize=8)
    axes[0].axvline(best_x, color='r', linestyle='--', label='Best')
    axes[0].set_xlabel(sweep_var)
    axes[0].set_ylabel('LAP Score (higher is better)')
    axes[0].set_title('Model Evidence (Laplace Approximation)')
    axes[0].set_xticks(xs)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, vaf, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].axvline(best_x, color='r', linestyle='--', label='Best')
    axes[1].set_xlabel(sweep_var)
    axes[1].set_ylabel('VAF (higher is better)')
    axes[1].set_title('Variance Accounted For')
    axes[1].set_xticks(xs)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"TMP sweep: {sweep_var} ({fixed_desc})", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{file_stem}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    svg_path = os.path.splitext(plot_path)[0] + '.svg'
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved to: {plot_path}")


def optimize_num_t_points(
        processed_data: List[np.ndarray],
        test_values: List[int],
        fixed_num_mps: int,
        adam_steps: int = 500,
        bfgs_steps: int = 10,
        save_dir: Optional[str] = None,
) -> Tuple[int, dict]:
    """Sweep num_t_points with num_MPs held fixed at fixed_num_mps."""
    results = {'num_t_points': [], 'lap_scores': [], 'vaf_scores': []}

    print(f"\n{'=' * 60}")
    print(f"SWEEP num_t_points (num_MPs fixed at {fixed_num_mps})")
    print(f"Testing: {test_values}")
    print(f"{'=' * 60}\n")

    for i, ntp in enumerate(test_values):
        print(f"[{i + 1}/{len(test_values)}] num_t_points = {ntp}")
        try:
            lap, vaf = _run_sweep(processed_data, fixed_num_mps, ntp, adam_steps, bfgs_steps)
        except Exception as e:
            print(f"  failed: {e}")
            lap, vaf = -np.inf, 0.0
        results['num_t_points'].append(ntp)
        results['lap_scores'].append(lap)
        results['vaf_scores'].append(vaf)
        print(f"  LAP={lap:.2f}  VAF={vaf:.4f}")

    best_idx = int(np.argmax(results['lap_scores']))
    best_num_t_points = results['num_t_points'][best_idx]
    print(f"\nBest num_t_points: {best_num_t_points}\n")

    if save_dir:
        _save_sweep_outputs(
            results=results,
            sweep_var='num_t_points',
            fixed_desc=f"num_MPs={fixed_num_mps}",
            save_dir=save_dir,
            file_stem='num_t_points_optimization',
        )

    return best_num_t_points, results


def optimize_num_mps(
        processed_data: List[np.ndarray],
        test_values: List[int],
        fixed_num_t_points: int,
        adam_steps: int = 500,
        bfgs_steps: int = 10,
        save_dir: Optional[str] = None,
) -> Tuple[int, dict]:
    """Sweep num_mps with num_t_points held fixed at fixed_num_t_points."""
    results = {'num_mps': [], 'lap_scores': [], 'vaf_scores': []}

    print(f"\n{'=' * 60}")
    print(f"SWEEP num_mps (num_t_points fixed at {fixed_num_t_points})")
    print(f"Testing: {test_values}")
    print(f"{'=' * 60}\n")

    for i, nm in enumerate(test_values):
        print(f"[{i + 1}/{len(test_values)}] num_mps = {nm}")
        try:
            lap, vaf = _run_sweep(processed_data, nm, fixed_num_t_points, adam_steps, bfgs_steps)
        except Exception as e:
            print(f"  failed: {e}")
            lap, vaf = -np.inf, 0.0
        results['num_mps'].append(nm)
        results['lap_scores'].append(lap)
        results['vaf_scores'].append(vaf)
        print(f"  LAP={lap:.2f}  VAF={vaf:.4f}")

    best_idx = int(np.argmax(results['lap_scores']))
    best_num_mps = results['num_mps'][best_idx]
    print(f"\nBest num_mps: {best_num_mps}\n")

    if save_dir:
        _save_sweep_outputs(
            results=results,
            sweep_var='num_mps',
            fixed_desc=f"num_t_points={fixed_num_t_points}",
            save_dir=save_dir,
            file_stem='num_mps_optimization',
        )

    return best_num_mps, results


def main() -> None:
    args = parse_args()

    # Fixed configuration (no CLI for these)
    data_dir = "../../data/pymotion_position_csv_files"
    tail_window = 50
    model_name_suffix: Optional[str] = None

    motion_ids, processed_segments, segment_motion_ids = process_motion_data(folder_path=data_dir,
                                                                             data_type="position",
                                                                             filtering= False)
    num_segments = len(processed_segments)
    print(f"Number of segments: {num_segments}")
    num_signals = processed_segments[0].shape[0]
    processed_data = processed_segments

#####################
    # # use mean subtracted segments instead full data
    # ms_segments, seg_means = subtract_segment_means(processed_segments)
    # processed_data = ms_segments
######################
    # Prepare save paths (now returns model_dir as well)
    model_name, model_path, model_dir = prepare_save_paths(args.num_mps,  args.num_t_points, model_name_suffix)
    optimization_dir = os.path.join(model_dir, 'TMP_optimization')

    # OPTIMIZATION MODE: sweep num_t_points and num_mps independently
    if args.optimize:
        os.makedirs(optimization_dir, exist_ok=True)

        best_ntp, _ = optimize_num_t_points(
            processed_data=processed_data,
            test_values=args.t_points_values,
            fixed_num_mps=args.num_mps,
            adam_steps=args.adam_steps,
            bfgs_steps=args.bfgs_steps,
            save_dir=optimization_dir,
        )

        best_nm, _ = optimize_num_mps(
            processed_data=processed_data,
            test_values=args.num_mps_values,
            fixed_num_t_points=args.num_t_points,
            adam_steps=args.adam_steps,
            bfgs_steps=args.bfgs_steps,
            save_dir=optimization_dir,
        )

        print(f"\nOptimization complete.")
        print(f"  Best num_t_points (num_MPs={args.num_mps}):    {best_ntp}")
        print(f"  Best num_mps     (num_t_points={args.num_t_points}): {best_nm}")
        print(f"  Outputs: {optimization_dir}")
        return

    # Initialize or load model
    model = initialize_or_load_model(
        num_t_points=args.num_t_points,
        num_mps=args.num_mps,
        num_segments=num_segments,
        num_signals=num_signals,
        processed_data=processed_data,
        use_pca=args.pca_init,
        model_path=model_path,
        load_existing=args.load,
    )

    if not args.load:
        # Train and persist
        train_and_save(
            model=model,
            processed_data=processed_data,
            model_path=model_path,
            adam_steps=args.adam_steps,
            bfgs_steps=args.bfgs_steps,
        )

    # Evaluate and plot artifacts (now with model_dir)
    evaluate_and_plot(
        model=model,
        processed_data=processed_data,
        model_name=model_name,
        model_dir=model_dir,
        tail_window=tail_window,
    )

    # Unified classification pipeline (TMP features)
    try:
        X_tmp = prepare_weights_for_classification(model, num_segments=num_segments, num_signals=num_signals, num_MPs=args.num_mps)
        y_tmp = np.array(segment_motion_ids)
        feature_names = [f"signal_{s}_mp_{m}" for s in range(num_signals) for m in range(args.num_mps)]
        cls_out_dir = os.path.join(model_dir, 'classification')
        run_classification_pipeline(
            X=X_tmp,
            y=y_tmp,
            out_dir=cls_out_dir,
            feature_names=feature_names,
            feature_structure={'n_signals': num_signals, 'n_features_per_signal': args.num_mps},
            primary_classifier='linear_svc',
            fixed_cm_vmin=0.0,
            fixed_cm_vmax=1.0,
            seed=42,
            cv_folds=5,perform_cv= True
        )
        print(f"  Classification artifacts saved to: {cls_out_dir}")
    except Exception as e:
        print(f"[warning] classification pipeline failed: {e}")

    print(f"\n✓ Experiment completed successfully!")
    print(f"  Model saved to: {model_path}")
    print(f"  Figures saved to: {os.path.join(model_dir, 'figures')}")


if __name__ == "__main__":
    main()