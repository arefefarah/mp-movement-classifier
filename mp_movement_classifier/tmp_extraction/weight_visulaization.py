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
import os
import torch
from pathlib import Path
from scipy import stats
import json
import argparse
import sys

from mp_movement_classifier.utils.utils import read_bvh_files, process_bvh_data,process_exp_map_data

JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

# CHANNEL_NAMES = []
# # Hip has 6 channels (position + rotation)
# CHANNEL_NAMES.extend([f'{JOINT_NAMES[0]}_Xpos', f'{JOINT_NAMES[0]}_Ypos', f'{JOINT_NAMES[0]}_Zpos',
#                       f'{JOINT_NAMES[0]}_Zrot', f'{JOINT_NAMES[0]}_Xrot', f'{JOINT_NAMES[0]}_Yrot'])
# # All other joints have 3 channels (Zrotation, Xrotation, Yrotation)
# for joint in JOINT_NAMES[1:]:
#     CHANNEL_NAMES.extend([f'{joint}_Zrot', f'{joint}_Xrot', f'{joint}_Yrot'])
# CHANNEL_NAMES = CHANNEL_NAMES[3:]
# print(len(CHANNEL_NAMES))

CHANNEL_NAMES = []
for joint in JOINT_NAMES:
    CHANNEL_NAMES.extend([f'{joint}_X', f'{joint}_Y', f'{joint}_Z'])

print(len(CHANNEL_NAMES))

COORD_NAMES = ['X', 'Y', 'Z']
DEFAULT_MODEL_DIR = "../../results/tmp_configs"
DEFAULT_DATA_DIR = "../../data/filtered_bvh_files"
DEFAULT_MOTION_MAPPING = "../../data/motion_mapping.json"


def load_motion_mapping(mapping_file):

    with open(mapping_file, 'r') as f:
        data = json.load(f)
        # Invert the mapping: from {name: id} to {id: name}
        if "mapping" in data:
            motion_mapping = data["mapping"]
            motion_id_to_name = {v: k for k, v in motion_mapping.items()}
        else:
            motion_id_to_name = {v: k for k, v in data.items()}
    print(f"✓ Loaded motion mapping with {len(motion_id_to_name)} motion types.")
    # for motion_id, name in sorted(motion_id_to_name.items()):
    #     print(f"    ID {motion_id}: {name}")
    return motion_id_to_name

def load_segment_motion_ids(bvh_dir, cutoff_freq=6.0):


        bvh_data, motion_ids = read_bvh_files(bvh_dir)

        processed_segments, segment_motion_ids = process_bvh_data(
            bvh_dir,
            motion_ids,
            cutoff_freq=cutoff_freq
        )

        print(f"loaded {len(segment_motion_ids)} segments")

        # Count segments per motion
        unique_motions, counts = np.unique(segment_motion_ids, return_counts=True)
        print(f"\nSegments per motion:")
        for motion_id, count in zip(unique_motions, counts):
            print(f"    Motion {motion_id}: {count} segments")

        return np.array(segment_motion_ids)


def compare_weights_across_movements(weights, motion_ids, motion_names_dict=None,save_dir='./plots'):
    """
    Compare MP weights across different movements for all joints.

    Args:
        weights: [num_segments, num_joints, num_coords, num_MPs]
        motion_ids: [num_segments] array of motion IDs
        motion_names_dict: dict mapping motion ID to motion name
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_segments, num_joints_coord, num_MPs = weights.shape

    unique_motions = np.unique(motion_ids)

    if motion_names_dict:
        motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
    else:
        motion_labels = [f'Motion {m}' for m in unique_motions]

    # Average weights across coordinates for each joint and MP
    # Shape: [num_segments, num_joints, num_MPs]
    # weights_per_joint = weights.mean(axis=2)

    # For each MP, show comparison across joints and movements
    for mp_idx in range(min(num_MPs, 10)):  # Show first 10 MPs
        n_motions = len(unique_motions)

        for i, motion_id in enumerate(unique_motions):
            mask = motion_ids == motion_id
            motion_weights = weights[mask, :, mp_idx]  # [n_segments, num_joints_coord]

            # Average across segments of this motion
            avg_weights = motion_weights.mean(axis=0)  # [num_joints_coord]
            std_weights = motion_weights.std(axis=0)

            plt.figure(figsize=(10, 6))
            plt.bar(CHANNEL_NAMES, avg_weights,
                   yerr=std_weights, width=0.5 ,  capsize=3,
                   color='steelblue', alpha=0.7)
            plt.ylabel(f'Average Weight ',fontsize=10)
            plt.xticks(rotation=45, ha='right',fontsize=6)  # 'ha' for horizontal alignment
            plt.xlabel('Channels')
            plt.title(f'MP {mp_idx + 1}  Weights for {motion_labels[i]}\n({mask.sum()} segments)',
                      fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/mp_{mp_idx + 1:02d}_{motion_id}_weights.png', dpi=150, bbox_inches='tight')
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
    ax1.set_xticklabels(CHANNEL_NAMES, rotation=45, ha='right', fontsize=9)
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
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR,
                        help=f'Directory containing trained models (default: {DEFAULT_MODEL_DIR})')
    parser.add_argument('--bvh-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Directory containing BVH files (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--output-dir', type=str, default='./weight_analysis',
                        help='Output directory for plots (default: ./weight_analysis)')
    parser.add_argument('--num-mps', type=int, default=20,
                        help='Number of MPs in model (default: 20)')
    parser.add_argument('--cutoff-freq', type=float, default=6.0,
                        help='Cutoff frequency used in training (default: 6.0)')

    args = parser.parse_args()

    # Load motion mapping
    motion_id_to_name = load_motion_mapping(DEFAULT_MOTION_MAPPING)

    # model_subdir = os.path.join(DEFAULT_MODEL_DIR, f"pos_filtered_mp_model_20_cutoff_3_tpoints_30")
    model_subdir = os.path.join(DEFAULT_MODEL_DIR, f"position_mp_model_5")
    # model_name = "mp_model_20_PC_init_cutoff_3_tpoints_30"
    model_name = "mp_model_5_PC_tpoints_30"

    model_path = os.path.join(model_subdir,model_name)

    # extract weights form the model
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    weights_list = []
    idx = 0
    while f"weights.{idx}" in model_data["model_state_dict"]:
        weights_list.append(model_data["model_state_dict"][f"weights.{idx}"].numpy())
        idx += 1
    weights = np.stack(weights_list, axis=0)

    args.bvh_dir = DEFAULT_DATA_DIR
    # segment_motion_ids = load_segment_motion_ids(args.bvh_dir, args.cutoff_freq)
    folder_path = "../../data/position_csv_files"
    motion_ids, processed_segments, segment_motion_ids = process_exp_map_data(folder_path=folder_path)

    # Run analyses
    print(f"\n{'=' * 70}")
    print("MP Weights Across Movements".center(70))
    print(f"{'=' * 70}")
    output_dir = os.path.join(model_subdir,"weights_analysis")
    compare_weights_across_movements(weights, segment_motion_ids,motion_id_to_name,
                                     output_dir)

    # print(f"\n{'=' * 70}")
    # print("ANALYSIS 2: Coordinate Variance Among Joints".center(70))
    # print(f"{'=' * 70}")
    # variance_results = analyze_coordinate_variance(
    #     weights, segment_motion_ids, motion_id_to_name,
    #     coord_idx=1, save_dir=output_dir
    # )

    # print(f"\n{'=' * 70}")
    # print(f"ANALYSIS 3: Joint-Specific coordinate -Variance & Significance".center(70))
    # print(f"{'=' * 70}")
    # avg_vars, std_vars, f_stat, p_value = analyze_joint_coord_variance(
    #     weights, segment_motion_ids, motion_id_to_name,coord_idx=1, save_dir=output_dir
    # )

if __name__ == "__main__":
    main()