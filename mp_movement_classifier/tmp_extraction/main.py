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
    parser.add_argument("--num-mps", type=int, default=20, help="Number of movement primitives.")
    parser.add_argument("--num-t-points", type=int, default=30, help="Number of time discretization points.")
    parser.add_argument("--cutoff-freq", type=float, default=3, help="Cutoff frequency for preprocessing.")
    parser.add_argument("--adam-steps", type=int, default=100, help="Number of ADAM optimization steps.")
    parser.add_argument("--bfgs-steps", type=int, default=30, help="Number of BFGS optimization steps.")
    parser.add_argument("--pca-init", action="store_true", default=True, help="Use PCA initialization.")
    parser.add_argument("--no-pca-init", dest="pca_init", action="store_false", help="Disable PCA initialization.")
    parser.add_argument("--load", action="store_true", help="Load an existing model if available.")
    parser.add_argument("--optimize-t-points", action="store_false",
                        help="Run optimization to find best num_t_points value.")
    parser.add_argument("--t-points-values", type=int, nargs='+', default=None,
                        help="Specific num_t_points values to test (e.g., --t-points-values 20 30 40 60)")

    return parser.parse_args()


def prepare_save_paths(num_mps: int, cutoff_freq: float, num_t_points: int, model_name_suffix: Optional[str]) -> Tuple[str, str, str]:
    """
    Prepare save paths for model and figures.
    """
    # model_dir = os.path.join("./../../results/tmp_configs",
    #                          f"new_seg_mp_model_{num_mps}_phase_three")
    model_dir = os.path.join("./../../results/tmp_configs", f"new_seg_mp_model_{num_mps}_tpoints_{num_t_points}_phase_three")
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


def optimize_num_t_points(
        processed_data: List[np.ndarray],
        num_mps: int,
        test_values: Optional[List[int]] = None,
        adam_steps: int = 500,
        bfgs_steps: int = 10,
        save_dir: Optional[str] = None,
) -> Tuple[int, dict]:
    """
    Find the optimal num_t_points by training models with different values
    and comparing their performance using Laplace approximation (model evidence)
    and VAF (Variance Accounted For).

    Args:
        processed_data: List of preprocessed motion segments
        num_mps: Number of movement primitives to use
        test_values: List of num_t_points values to test. If None, uses heuristic values
        adam_steps: Number of ADAM optimization steps
        bfgs_steps: Number of BFGS optimization steps
        model_dir: Directory to save optimization results

    Returns:
        best_num_t_points: The optimal value
        results: Dictionary containing all results for analysis
    """

    # Setup
    num_segments = len(processed_data)
    num_signals = processed_data[0].shape[0]
    segment_lengths = np.array([seg.shape[1] for seg in processed_data])

    # Determine test values based on data characteristics if not provided
    if test_values is None:
        median_length = int(np.median(segment_lengths))
        test_values = [
            median_length // 4,  # Very coarse
            median_length // 2,  # Coarse
            median_length,  # Matched to median
            median_length * 2,  # Fine
            median_length * 3,  # Very fine
        ]
        print(f"Auto-selected test values based on median segment length ({median_length}): {test_values}")

    # Storage for results
    results = {
        'num_t_points': [],
        'lap_scores': [],
        'vaf_scores': [],
        'final_log_p': [],
        'convergence_steps': [],
    }

    print(f"\n{'=' * 60}")
    print(f"OPTIMIZING num_t_points")
    print(f"{'=' * 60}")
    print(f"Testing values: {test_values}")
    print(f"Number of MPs: {num_mps}")
    print(f"Number of segments: {num_segments}")
    print(f"Segment length range: [{segment_lengths.min()}, {segment_lengths.max()}]")
    print(f"{'=' * 60}\n")

    for i, num_t_points in enumerate(test_values):
        print(f"\n[{i + 1}/{len(test_values)}] Testing num_t_points = {num_t_points}")
        print("-" * 50)

        try:
            # Initialize model with PCA
            print(f"  Initializing model...")
            model = MP_model(
                num_t_points=num_t_points,
                num_MPs=num_mps,
                num_segments=num_segments,
                num_signals=num_signals,
                init_data=processed_data,
            )

            # Train model
            print(f"  Training (ADAM={adam_steps}, BFGS={bfgs_steps})...")
            model.learn(processed_data, adam_steps=adam_steps, bfgs_steps=bfgs_steps)

            # Extract metrics
            final_vaf = model.VAF_curve[-1]
            final_log_p = model.learn_curve[-1].item() if torch.is_tensor(model.learn_curve[-1]) else model.learn_curve[
                -1]
            convergence_steps = len(model.learn_curve)

            print(f"  Computing Laplace approximation...")
            lap_score = model.Laplace_approx(processed_data)

            # Store results
            results['num_t_points'].append(num_t_points)
            results['lap_scores'].append(lap_score)
            results['vaf_scores'].append(final_vaf)
            results['final_log_p'].append(final_log_p)
            results['convergence_steps'].append(convergence_steps)

            print(f"  ✓ Results:")
            print(f"    - LAP score: {lap_score:.2f}")
            print(f"    - VAF: {final_vaf:.4f}")
            print(f"    - Final log P: {final_log_p:.2f}")
            print(f"    - Convergence steps: {convergence_steps}")

        except Exception as e:
            print(f"  ✗ Failed with error: {e}")
            results['num_t_points'].append(num_t_points)
            results['lap_scores'].append(-np.inf)
            results['vaf_scores'].append(0.0)
            results['final_log_p'].append(-np.inf)
            results['convergence_steps'].append(0)

    # Find best num_t_points based on LAP score (model evidence)
    best_idx = np.argmax(results['lap_scores'])
    best_num_t_points = results['num_t_points'][best_idx]

    print(f"\n{'=' * 60}")
    print(f"OPTIMIZATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\nComparison Table:")
    print(f"{'num_t_points':<15} {'LAP Score':<15} {'VAF':<10} {'Log P':<15} {'Steps':<10}")
    print("-" * 65)

    for i in range(len(results['num_t_points'])):
        marker = " ← BEST" if i == best_idx else ""
        print(f"{results['num_t_points'][i]:<15} "
              f"{results['lap_scores'][i]:<15.2f} "
              f"{results['vaf_scores'][i]:<10.4f} "
              f"{results['final_log_p'][i]:<15.2f} "
              f"{results['convergence_steps'][i]:<10}{marker}")

    print(f"\n{'=' * 60}")
    print(f"BEST num_t_points: {best_num_t_points}")
    print(f"  - LAP Score: {results['lap_scores'][best_idx]:.2f}")
    print(f"  - VAF: {results['vaf_scores'][best_idx]:.4f}")
    print(f"{'=' * 60}\n")

    # Save results if directory provided
    if save_dir:
        results_dir = os.path.join(save_dir, 'optimization')
        os.makedirs(results_dir, exist_ok=True)

        # Save numerical results
        results_file = os.path.join(results_dir, 'num_t_points_optimization.txt')
        with open(results_file, 'w') as f:
            f.write(f"Optimization Results for num_t_points\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"{'num_t_points':<15} {'LAP Score':<15} {'VAF':<10} {'Log P':<15} {'Steps':<10}\n")
            f.write("-" * 65 + "\n")
            for i in range(len(results['num_t_points'])):
                marker = " ← BEST" if i == best_idx else ""
                f.write(f"{results['num_t_points'][i]:<15} "
                        f"{results['lap_scores'][i]:<15.2f} "
                        f"{results['vaf_scores'][i]:<10.4f} "
                        f"{results['final_log_p'][i]:<15.2f} "
                        f"{results['convergence_steps'][i]:<10}{marker}\n")
            f.write(f"\nBest num_t_points: {best_num_t_points}\n")

        print(f"Results saved to: {results_file}")

        # Create visualization plot
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('num_t_points Optimization Results', fontsize=16)

            # LAP scores
            axes[0, 0].plot(results['num_t_points'], results['lap_scores'], 'o-', linewidth=2, markersize=8)
            axes[0, 0].axvline(best_num_t_points, color='r', linestyle='--', label='Best')
            axes[0, 0].set_xlabel('num_t_points')
            axes[0, 0].set_ylabel('LAP Score (higher is better)')
            axes[0, 0].set_title('Model Evidence (Laplace Approximation)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # VAF scores
            axes[0, 1].plot(results['num_t_points'], results['vaf_scores'], 'o-', linewidth=2, markersize=8,
                            color='green')
            axes[0, 1].axvline(best_num_t_points, color='r', linestyle='--', label='Best')
            axes[0, 1].set_xlabel('num_t_points')
            axes[0, 1].set_ylabel('VAF (higher is better)')
            axes[0, 1].set_title('Variance Accounted For')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Log P
            axes[1, 0].plot(results['num_t_points'], results['final_log_p'], 'o-', linewidth=2, markersize=8,
                            color='orange')
            axes[1, 0].axvline(best_num_t_points, color='r', linestyle='--', label='Best')
            axes[1, 0].set_xlabel('num_t_points')
            axes[1, 0].set_ylabel('Final Log P (higher is better)')
            axes[1, 0].set_title('Final Joint Probability')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Convergence steps
            axes[1, 1].plot(results['num_t_points'], results['convergence_steps'], 'o-', linewidth=2, markersize=8,
                            color='purple')
            axes[1, 1].axvline(best_num_t_points, color='r', linestyle='--', label='Best')
            axes[1, 1].set_xlabel('num_t_points')
            axes[1, 1].set_ylabel('Convergence Steps')
            axes[1, 1].set_title('Training Convergence')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = os.path.join(results_dir, 'num_t_points_optimization.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to: {plot_file}")

        except Exception as e:
            print(f"Warning: Could not create visualization plot: {e}")

    return best_num_t_points, results


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
    model_name, model_path, model_dir = prepare_save_paths(args.num_mps, args.cutoff_freq, args.num_t_points, model_name_suffix)

    # OPTIMIZATION MODE: Find best num_t_points
    args.optimize_t_points = False
    if args.optimize_t_points:
        print("\n" + "=" * 60)
        print("RUNNING num_t_points OPTIMIZATION")
        print("=" * 60 + "\n")

        best_num_t_points, opt_results = optimize_num_t_points(
            processed_data=processed_data,
            num_mps=args.num_mps,
            test_values=args.t_points_values,
            adam_steps=args.adam_steps,
            bfgs_steps=args.bfgs_steps,
            save_dir=model_dir,
        )

        print(f"\n✓ Optimization completed!")
        print(f"  Recommended num_t_points: {best_num_t_points}")
        print(f"  Use this value with: --num-t-points {best_num_t_points}")

        return  # Exit after optimization

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

    # if not args.load:
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
            also_run_random_forest=True,
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