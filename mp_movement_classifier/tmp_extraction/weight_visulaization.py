#!/usr/bin/env python3
"""
Analyze and visualize TMP model weights across movements and joints.

This script integrates with your BVH processing pipeline to:
1. Load trained TMP model with segment-to-motion mapping
2. Compare weights across different movements
3. Analyze variance of coordinates among joints
4. Test statistical significance

Author: Adapted for BVH-based TMP model analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from scipy import stats
import json
import argparse
import sys

# Joint names (17 joints from H36M skeleton)
JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

COORD_NAMES = ['X', 'Y', 'Z']

# Default paths - EDIT THESE TO MATCH YOUR SETUP
DEFAULT_MODEL_DIR = "../../results/tmp_configs"
DEFAULT_DATA_DIR = "../../data/bvh_files"
DEFAULT_MOTION_MAPPING = "../../data/motion_mapping.json"


def load_motion_mapping(mapping_file):
    """
    Load motion ID to motion name mapping from JSON file.

    Returns:
        motion_id_to_name: dict mapping motion IDs to names
    """
    if not Path(mapping_file).exists():
        print(f"⚠️  Warning: Motion mapping file not found: {mapping_file}")
        return None

    try:
        with open(mapping_file, 'r') as f:
            data = json.load(f)
            # Invert the mapping: from {name: id} to {id: name}
            if "mapping" in data:
                motion_mapping = data["mapping"]
                motion_id_to_name = {v: k for k, v in motion_mapping.items()}
            else:
                motion_id_to_name = {v: k for k, v in data.items()}

        print(f"✓ Loaded motion mapping with {len(motion_id_to_name)} motion types:")
        for motion_id, name in sorted(motion_id_to_name.items()):
            print(f"    ID {motion_id}: {name}")

        return motion_id_to_name
    except Exception as e:
        print(f"❌ Error loading motion mapping: {e}")
        return None


def load_model_weights_and_metadata(model_path):
    """
    Load weights from trained TMP model along with metadata.

    Returns:
        weights_reshaped: [num_segments, num_joints, num_coords, num_MPs]
        saved_data: full model data
    """
    saved_data = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract weights from state dict
    weights_list = []
    idx = 0
    while f"weights.{idx}" in saved_data["model_state_dict"]:
        weights_list.append(saved_data["model_state_dict"][f"weights.{idx}"].numpy())
        idx += 1

    # Stack into [num_segments, num_signals, num_MPs]
    weights = np.stack(weights_list, axis=0)

    num_segments, num_signals, num_MPs = weights.shape
    print(f"\n{'=' * 60}")
    print(f"MODEL INFORMATION")
    print(f"{'=' * 60}")
    print(f"Number of segments: {num_segments}")
    print(f"Number of signals:  {num_signals}")
    print(f"Number of MPs:      {num_MPs}")
    print(f"Number of joints:   {num_signals // 3}")

    # Reshape to [num_segments, num_joints, num_coords, num_MPs]
    num_joints = num_signals // 3
    weights_reshaped = weights.reshape(num_segments, num_joints, 3, num_MPs)

    return weights_reshaped, saved_data


def load_segment_motion_ids(bvh_dir, cutoff_freq=6.0):
    """
    Process BVH files to get segment-to-motion mapping.
    This replicates the process_bvh_data workflow.

    Returns:
        segment_motion_ids: array of motion IDs for each segment
    """
    try:
        # Import required functions from your utils
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mp_movement_classifier.utils.utils import read_bvh_files, process_bvh_data

        print(f"\n{'=' * 60}")
        print(f"PROCESSING BVH DATA")
        print(f"{'=' * 60}")
        print(f"Reading BVH files from: {bvh_dir}")

        # Read BVH files
        bvh_data, motion_ids = read_bvh_files(bvh_dir)
        print(f"Loaded {len(bvh_data)} BVH files")

        # Process and segment
        print(f"Processing with cutoff frequency: {cutoff_freq} Hz")
        processed_segments, segment_motion_ids = process_bvh_data(
            bvh_data,
            motion_ids,
            cutoff_freq=cutoff_freq
        )

        print(f"Created {len(segment_motion_ids)} segments")

        # Count segments per motion
        unique_motions, counts = np.unique(segment_motion_ids, return_counts=True)
        print(f"\nSegments per motion:")
        for motion_id, count in zip(unique_motions, counts):
            print(f"    Motion {motion_id}: {count} segments")

        return np.array(segment_motion_ids)

    except ImportError as e:
        print(f"⚠️  Warning: Could not import processing functions: {e}")
        print(f"    Make sure mp_movement_classifier is in your Python path")
        return None
    except Exception as e:
        print(f"❌ Error processing BVH data: {e}")
        return None


def find_model_file(model_dir, num_mps=20, cutoff_freq=6.0, num_t_points=30):
    """
    Find model file based on naming convention from training script.

    Returns path to model file or None if not found.
    """
    model_subdir = Path(model_dir) / f"new_seg_mp_model_{num_mps}_cutoff_{cutoff_freq}_tpoints_{num_t_points}"

    if not model_subdir.exists():
        print(f"⚠️  Model directory not found: {model_subdir}")
        return None

    model_name = f"mp_model_{num_mps}_PC_init_cutoff_{cutoff_freq}_tpoints_{num_t_points}"
    model_path = model_subdir / model_name

    if not model_path.exists():
        print(f"⚠️  Model file not found: {model_path}")
        # Try to find any .pth file in the directory
        pth_files = list(model_subdir.glob("*.pth"))
        if pth_files:
            print(f"    Found alternative model file: {pth_files[0]}")
            return str(pth_files[0])
        return None

    return str(model_path)


# ============================================================================
# ANALYSIS 1: Compare weights of MPs across movements for all joints
# ============================================================================

def compare_weights_across_movements(weights, motion_ids, motion_names_dict=None, save_dir='./plots'):
    """
    Compare MP weights across different movements for all joints.

    Args:
        weights: [num_segments, num_joints, num_coords, num_MPs]
        motion_ids: [num_segments] array of motion IDs
        motion_names_dict: dict mapping motion ID to motion name
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_segments, num_joints, num_coords, num_MPs = weights.shape

    unique_motions = np.unique(motion_ids)

    # Get motion names
    if motion_names_dict:
        motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
    else:
        motion_labels = [f'Motion {m}' for m in unique_motions]

    # Average weights across coordinates for each joint and MP
    # Shape: [num_segments, num_joints, num_MPs]
    weights_per_joint = weights.mean(axis=2)

    # For each MP, show comparison across joints and movements
    for mp_idx in range(min(num_MPs, 10)):  # Show first 10 MPs
        n_motions = len(unique_motions)
        fig, axes = plt.subplots(1, n_motions,
                                 figsize=(5 * n_motions, 6),
                                 squeeze=False)

        for i, motion_id in enumerate(unique_motions):
            mask = motion_ids == motion_id
            motion_weights = weights_per_joint[mask, :, mp_idx]  # [n_segments, num_joints]

            # Average across segments of this motion
            avg_weights = motion_weights.mean(axis=0)  # [num_joints]
            std_weights = motion_weights.std(axis=0)

            ax = axes[0, i]
            ax.bar(range(num_joints), avg_weights,
                   yerr=std_weights, capsize=3,
                   color='steelblue', alpha=0.7)
            ax.set_xticks(range(num_joints))
            ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Average Weight', fontsize=10)
            ax.set_title(f'{motion_labels[i]}\n({mask.sum()} segments)', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle(f'MP {mp_idx + 1} Weights Across Joints and Movements',
                     fontsize=14, y=1.02, weight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/mp_{mp_idx + 1:02d}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ Saved {min(num_MPs, 10)} MP comparison plots to {save_dir}")


# ============================================================================
# ANALYSIS 2: Variance of features (one coordinate) among joints per movement
# ============================================================================

def analyze_coordinate_variance(weights, motion_ids, motion_names_dict=None,
                                coord_idx=0, save_dir='./plots'):
    """
    Analyze variance of one coordinate (default: Z) among all joints for each movement.

    Args:
        weights: [num_segments, num_joints, num_coords, num_MPs]
        motion_ids: [num_segments] array of motion IDs
        coord_idx: 0=X, 1=Y, 2=Z
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_segments, num_joints, num_coords, num_MPs = weights.shape
    coord_name = COORD_NAMES[coord_idx]

    unique_motions = np.unique(motion_ids)

    # Get motion names
    if motion_names_dict:
        motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
    else:
        motion_labels = [f'Motion {m}' for m in unique_motions]

    # Extract weights for the specified coordinate
    coord_weights = weights[:, :, coord_idx, :]  # [num_segments, num_joints, num_MPs]

    results = {}

    for motion_id, motion_label in zip(unique_motions, motion_labels):
        mask = motion_ids == motion_id
        motion_weights = coord_weights[mask]  # [n_segs, num_joints, num_MPs]

        # For each MP, calculate variance across joints
        variances_per_mp = []
        for mp_idx in range(num_MPs):
            mp_weights = motion_weights[:, :, mp_idx]  # [n_segs, num_joints]
            # Variance across joints for each segment, then average
            var_across_joints = np.var(mp_weights, axis=1).mean()
            variances_per_mp.append(var_across_joints)

        results[motion_id] = {
            'label': motion_label,
            'variances': np.array(variances_per_mp),
            'mean_variance': np.mean(variances_per_mp),
            'std_variance': np.std(variances_per_mp)
        }

    # Plot variance across MPs for each movement
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(num_MPs)
    width = 0.8 / len(unique_motions)

    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_motions)))

    for i, motion_id in enumerate(unique_motions):
        variances = results[motion_id]['variances']
        label = results[motion_id]['label']
        offset = (i - len(unique_motions) / 2) * width + width / 2
        ax.bar(x + offset, variances, width,
               label=label, alpha=0.8, color=colors[i])

    ax.set_xlabel('Movement Primitive', fontsize=12, weight='bold')
    ax.set_ylabel(f'Variance of {coord_name}-coordinate Across Joints', fontsize=12, weight='bold')
    ax.set_title(f'Variance of {coord_name}-coordinate Among All Joints per Movement',
                 fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'MP{i + 1}' for i in range(num_MPs)], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/variance_{coord_name}_across_joints.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"VARIANCE ANALYSIS: {coord_name}-coordinate across joints")
    print(f"{'=' * 60}")
    for motion_id in unique_motions:
        stats_data = results[motion_id]
        print(f"\n{stats_data['label']}:")
        print(f"  Mean variance: {stats_data['mean_variance']:.6f}")
        print(f"  Std variance:  {stats_data['std_variance']:.6f}")

    print(f"\n✓ Saved variance analysis plot to {save_dir}")

    return results


# ============================================================================
# ANALYSIS 3: Average Z-variance for each joint & significance test
# ============================================================================

def analyze_joint_coord_variance(weights, motion_ids, motion_names_dict=None, coord_idx=0,save_dir='./plots'):
    """
    Calculate average Z-coordinate variance for each joint across MPs.
    Test statistical significance using one-way ANOVA.

    Args:
        weights: [num_segments, num_joints, num_coords, num_MPs]
        motion_ids: [num_segments] array of motion IDs
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_segments, num_joints, num_coords, num_MPs = weights.shape

    # Extract Z-coordinate weights: [num_segments, num_joints, num_MPs]
    z_weights = weights[:, :, coord_idx, :]

    # Calculate variance of each joint across MPs (for each segment)
    joint_variances = []
    for joint_idx in range(num_joints):
        joint_z = z_weights[:, joint_idx, :]  # [num_segments, num_MPs]
        # Variance across MPs for each segment
        var_per_segment = np.var(joint_z, axis=1)  # [num_segments]
        joint_variances.append(var_per_segment)

    joint_variances = np.array(joint_variances)  # [num_joints, num_segments]

    # Average variance for each joint
    avg_joint_variances = joint_variances.mean(axis=1)  # [num_joints]
    std_joint_variances = joint_variances.std(axis=1)

    # Statistical significance test: Are variances significantly different across joints?
    # Use one-way ANOVA
    f_stat, p_value = stats.f_oneway(*[joint_variances[i] for i in range(num_joints)])

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))

    # Subplot 1: Bar plot of average variance per joint
    ax1 = plt.subplot(1, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, num_joints))
    bars = ax1.bar(range(num_joints), avg_joint_variances,
                   yerr=std_joint_variances,
                   color=colors, alpha=0.7, capsize=5)
    ax1.set_xticks(range(num_joints))
    ax1.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Average Z-coordinate Variance', fontsize=11, weight='bold')
    ax1.set_title('Average Z-Variance Among MPs\nfor Each Joint', fontsize=12, weight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add significance annotation
    sig_text = f'ANOVA: F={f_stat:.2f}, p={p_value:.2e}\n' + \
               ('**SIGNIFICANT**' if p_value < 0.05 else 'Not Significant')
    ax1.text(0.98, 0.98, sig_text,
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=10, weight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Subplot 2: Box plot showing distribution of variances
    ax2 = plt.subplot(1, 3, 2)
    bp = ax2.boxplot([joint_variances[i] for i in range(num_joints)],
                     labels=JOINT_NAMES, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel(f'{COORD_NAMES[coord_idx]}-coordinate Variance Distribution', fontsize=11, weight='bold')
    ax2.set_title(f'Distribution of {COORD_NAMES[coord_idx]}-Variance\nfor Each Joint', fontsize=12, weight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Subplot 3: Heatmap of variance per joint per motion
    ax3 = plt.subplot(1, 3, 3)
    unique_motions = np.unique(motion_ids)
    variance_matrix = np.zeros((len(unique_motions), num_joints))

    for i, motion_id in enumerate(unique_motions):
        mask = motion_ids == motion_id
        motion_z_weights = z_weights[mask]  # [n_segs, num_joints, num_MPs]
        for j in range(num_joints):
            joint_z = motion_z_weights[:, j, :]
            variance_matrix[i, j] = np.var(joint_z, axis=1).mean()

    im = ax3.imshow(variance_matrix, aspect='auto', cmap='YlOrRd')
    ax3.set_xticks(range(num_joints))
    ax3.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=9)
    ax3.set_yticks(range(len(unique_motions)))

    if motion_names_dict:
        motion_labels = [motion_names_dict.get(m, f'Motion {m}')[:15]
                         for m in unique_motions]
    else:
        motion_labels = [f'Motion {m}' for m in unique_motions]
    ax3.set_yticklabels(motion_labels, fontsize=9)
    ax3.set_title(f'{COORD_NAMES[coord_idx]}-Variance Heatmap\n(Motion × Joint)', fontsize=12, weight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Variance', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/joint_{COORD_NAMES[coord_idx]}_variance_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print detailed results
    print(f"\n{'=' * 60}")
    print(f"Z-VARIANCE ANALYSIS FOR EACH JOINT")
    print(f"{'=' * 60}")
    print(f"\n{'Joint':<15} {'Avg Variance':<15} {'Std Dev':<15}")
    print(f"{'-' * 45}")
    for i in range(num_joints):
        print(f"{JOINT_NAMES[i]:<15} {avg_joint_variances[i]:<15.6f} {std_joint_variances[i]:<15.6f}")

    print(f"\n{'=' * 60}")
    print(f"STATISTICAL SIGNIFICANCE TEST (One-Way ANOVA)")
    print(f"{'=' * 60}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value:     {p_value:.2e}")
    print(f"Result:      {'**SIGNIFICANT** (p < 0.05)' if p_value < 0.05 else 'NOT SIGNIFICANT (p >= 0.05)'}")
    print(f"\nInterpretation: The Z-coordinate variances across joints are")
    print(f"                {'statistically different' if p_value < 0.05 else 'not statistically different'}")

    print(f"\n✓ Saved joint Z-variance analysis to {save_dir}")

    return avg_joint_variances, std_joint_variances, f_stat, p_value


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze TMP model weights across movements and joints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect model and process BVH data
  python %(prog)s --auto

  # Specify custom paths
  python %(prog)s --model-dir /path/to/models --bvh-dir /path/to/bvh

  # Use specific model file
  python %(prog)s --model-path /path/to/model.pth --bvh-dir /path/to/bvh
        """
    )

    parser.add_argument('--auto', action='store_true',
                        help='Auto-detect model and BVH data using default paths')
    parser.add_argument('--model-path', type=str,
                        help='Path to trained model file (overrides --model-dir)')
    parser.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR,
                        help=f'Directory containing trained models (default: {DEFAULT_MODEL_DIR})')
    parser.add_argument('--bvh-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Directory containing BVH files (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--motion-mapping', type=str, default=DEFAULT_MOTION_MAPPING,
                        help=f'Path to motion mapping JSON (default: {DEFAULT_MOTION_MAPPING})')
    parser.add_argument('--output-dir', type=str, default='./weight_analysis',
                        help='Output directory for plots (default: ./weight_analysis)')
    parser.add_argument('--num-mps', type=int, default=20,
                        help='Number of MPs in model (default: 20)')
    parser.add_argument('--cutoff-freq', type=float, default=6.0,
                        help='Cutoff frequency used in training (default: 6.0)')
    parser.add_argument('--num-t-points', type=int, default=30,
                        help='Number of time points in model (default: 30)')

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("TMP MODEL WEIGHT ANALYSIS".center(70))
    print(f"{'=' * 70}\n")

    # Load motion mapping
    motion_id_to_name = load_motion_mapping(args.motion_mapping)

    # Find or use specified model file
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_model_file(
            args.model_dir,
            args.num_mps,
            args.cutoff_freq,
            args.num_t_points
        )
        if model_path is None:
            print(f"\n❌ Could not find model file. Please specify --model-path")
            return

    print(f"\nUsing model: {model_path}")

    # Load model weights
    print(f"\nLoading model weights...")
    weights, model_data = load_model_weights_and_metadata(model_path)

    # Process BVH data to get segment motion IDs
    segment_motion_ids = load_segment_motion_ids(args.bvh_dir, args.cutoff_freq)

    if segment_motion_ids is None:
        print(f"\n⚠️  Warning: Could not load segment motion IDs from BVH data")
        print(f"    Creating dummy labels (all segments = one group)")
        segment_motion_ids = np.zeros(weights.shape[0], dtype=int)
    elif len(segment_motion_ids) != weights.shape[0]:
        print(f"\n⚠️  Warning: Segment count mismatch!")
        print(f"    Model has {weights.shape[0]} segments")
        print(f"    BVH data has {len(segment_motion_ids)} segments")
        print(f"    Using min({weights.shape[0]}, {len(segment_motion_ids)}) segments")
        min_segments = min(weights.shape[0], len(segment_motion_ids))
        weights = weights[:min_segments]
        segment_motion_ids = segment_motion_ids[:min_segments]

    # Run analyses
    print(f"\n{'=' * 70}")
    print("ANALYSIS 1: MP Weights Across Movements and Joints".center(70))
    print(f"{'=' * 70}")
    output_dir = "../../results/tmp_configs/new_seg_mp_model_20_cutoff_6.0_tpoints_30/weights_analysis"
    compare_weights_across_movements(weights, segment_motion_ids,
                                     motion_id_to_name, output_dir)

    print(f"\n{'=' * 70}")
    print("ANALYSIS 2: Coordinate Variance Among Joints".center(70))
    print(f"{'=' * 70}")
    variance_results = analyze_coordinate_variance(
        weights, segment_motion_ids, motion_id_to_name,
        coord_idx=1, save_dir=output_dir
    )

    print(f"\n{'=' * 70}")
    print(f"ANALYSIS 3: Joint-Specific coordinate -Variance & Significance".center(70))
    print(f"{'=' * 70}")
    avg_vars, std_vars, f_stat, p_value = analyze_joint_coord_variance(
        weights, segment_motion_ids, motion_id_to_name,coord_idx=1, save_dir=output_dir
    )



if __name__ == "__main__":
    main()