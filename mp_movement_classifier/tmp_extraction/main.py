import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from TMP_model import MP_model
from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_bvh_data,
    read_bvh_files,
    save_model_with_full_state,

)
from mp_movement_classifier.utils.plotting import (
    plot_learn_curve, plot_mp,
    plot_reconstructions,
    set_figures_directory
)
from mp_movement_classifier.utils import config

DEFAULT_DATA_DIR = "../../data/bvh_files"
DEFAULT_TAIL_WINDOW = 50
MODEL_NAME_SUFFIX: Optional[str] = None


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
    return parser.parse_args()


def prepare_save_paths(num_mps: int, cutoff_freq: float, num_t_points: int, model_name_suffix: Optional[str]) -> Tuple[str, str, str]:
    """
    Prepare save paths for model and figures.
    """
    # Organize outputs under a dedicated directory
    model_dir = os.path.join(config.SAVING_DIR, f"test_seg_mp_model_{num_mps}_cutoff_{cutoff_freq}_tpoints_{num_t_points}")
    os.makedirs(model_dir, exist_ok=True)

    model_name = f"mp_model_{num_mps}_PC_init_cutoff_{cutoff_freq}_tpoints_{num_t_points}"
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


def main() -> None:
    args = parse_args()

    # Fixed configuration (no CLI for these)
    data_dir = DEFAULT_DATA_DIR
    tail_window = DEFAULT_TAIL_WINDOW
    model_name_suffix = MODEL_NAME_SUFFIX

    # Load BVH data
    print(f"Reading BVH files from: {data_dir}")
    bvh_data, motion_ids = read_bvh_files(data_dir)

    # Preprocess data including the segmentation step
    # print(f"Processing BVH data with cutoff frequency: {args.cutoff_freq}")
    processed_data, segment_motion_ids = process_bvh_data(
        data_dir,
        motion_ids,
        cutoff_freq=args.cutoff_freq,
    )

    if not processed_data:
        raise RuntimeError("No processed data segments found. Check the input directory and preprocessing parameters.")

    num_segments = len(processed_data)
    num_signals = processed_data[0].shape[0]
    print(f"num of segments : {num_segments}")
    print(f"num of signal   : {num_signals}")

    # Prepare save paths (now returns model_dir as well)
    model_name, model_path, model_dir = prepare_save_paths(args.num_mps, args.cutoff_freq, args.num_t_points, model_name_suffix)

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

    print(f"\n✓ Experiment completed successfully!")
    print(f"  Model saved to: {model_path}")
    print(f"  Figures saved to: {os.path.join(model_dir, 'figures')}")


if __name__ == "__main__":
    main()