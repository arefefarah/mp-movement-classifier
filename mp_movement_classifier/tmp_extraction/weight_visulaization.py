#!/usr/bin/env python3
"""
Analyze and visualize TMP model weights across movements and joints.

This script integrates with your BVH processing pipeline to:
1. Load trained TMP model with segment-to-motion mapping
2. Compare weights across different movements
3. Analyze variance of coordinates among joints
4. Test statistical significance

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

from mp_movement_classifier.utils.utils import read_bvh_files, process_bvh_data,process_motion_data

JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]
CHANNEL_NAMES = []
for joint in JOINT_NAMES:
    CHANNEL_NAMES.extend([f'{joint}_Xpos', f'{joint}_Ypos', f'{joint}_Zpos'])

#### for quaternion
# JOINT_NAMES = [
#     'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
#     'Spine', 'Thorax', 'Neck', 'Head',
#     'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
# ]
# COORD_NAMES = ['W','X', 'Y', 'Z']
# CHANNEL_NAMES = []
# for joint in JOINT_NAMES:
#     if joint=='Hip':
#         CHANNEL_NAMES.extend([f'{joint}_Xpos', f'{joint}_Ypos', f'{joint}_Zpos'])
#     for coord in COORD_NAMES:
#         CHANNEL_NAMES.extend([f'{joint}_{coord}'])

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


def weights_barplot_across_channels(weights, motion_ids, motion_names_dict=None, save_dir='./plots'):
    """
    Plot weight distributions across channels.

    For each channel, creates one figure showing the first 5 MP weights
    grouped by movement. All 5 MPs of one movement share the same color.
    Colors are consistent across all channels based on motion ID.

    Parameters:
    -----------
    weights : ndarray, shape (num_segments, num_joints_coord, num_MPs)
        Weight array
    motion_ids : ndarray, shape (num_segments,)
        Motion identifier for each segment
    motion_names_dict : dict, optional
        Mapping from motion_id to readable names
    save_dir : str
        Directory to save plots
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_segments, num_joints_coord, num_MPs = weights.shape
    unique_motions = np.unique(motion_ids)
    n_motions = len(unique_motions)

    # Create motion labels
    if motion_names_dict:
        motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
    else:
        motion_labels = [f'Motion {m}' for m in unique_motions]

    fixed_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#aec7e8',  # Light blue
        '#ffbb78',  # Light orange
        '#98df8a',  # Light green
        '#ff9896',  # Light red
        '#c5b0d5',  # Light purple
    ]

    # Create color mapping: motion_id -> color
    # This ensures each motion_id always gets the same color
    color_map = {}
    for idx, motion_id in enumerate(unique_motions):
        color_map[motion_id] = fixed_colors[idx % len(fixed_colors)]

    # Number of MPs to display
    n_mps_to_show = min(5, num_MPs)

    # Iterate over each channel
    for joint_idx in range(num_joints_coord):
        # Increase figure size to accommodate legend outside
        fig, ax = plt.subplots(figsize=(18, 8))

        # Calculate positions and data for all movements
        all_avg_weights = []
        all_std_weights = []
        x_positions = []
        bar_colors = []
        x_labels = []

        current_x = 0

        for motion_idx, motion_id in enumerate(unique_motions):
            # Get all segments for this motion
            mask = motion_ids == motion_id
            motion_weights = weights[mask, joint_idx, :n_mps_to_show]  # shape: (n_segments, 5)

            # Calculate statistics across segments
            avg_weights = motion_weights.mean(axis=0)  # shape: (5,)
            std_weights = motion_weights.std(axis=0)  # shape: (5,)

            all_avg_weights.extend(avg_weights)
            all_std_weights.extend(std_weights)

            # Create x positions for this movement's 5 MPs (grouped together)
            movement_x_positions = np.arange(current_x, current_x + n_mps_to_show)
            x_positions.extend(movement_x_positions)

            # Use the fixed color from color_map for this motion_id
            motion_color = color_map[motion_id]
            bar_colors.extend([motion_color] * n_mps_to_show)

            # Create labels for x-axis
            for mp_idx in range(n_mps_to_show):
                x_labels.append(f'MP{mp_idx + 1}')

            # Move to next movement (add a gap)
            current_x += n_mps_to_show + 1  # +1 for spacing between movements

        # Plot all bars
        bars = ax.bar(x_positions, all_avg_weights,
                      yerr=all_std_weights, capsize=4,
                      color=bar_colors, alpha=0.75,
                      edgecolor='black', linewidth=0.5,
                      width=0.8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
        ax.set_xlabel('Movement Primitives (grouped by movement)', fontsize=13, fontweight='bold')

        # Add vertical separator lines between movements
        for motion_idx in range(1, n_motions):
            separator_x = motion_idx * (n_mps_to_show + 1) - 0.5
            ax.axvline(x=separator_x, color='gray', linestyle='--',
                       linewidth=1.5, alpha=0.5)

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        for motion_idx, (motion_id, motion_label) in enumerate(zip(unique_motions, motion_labels)):
            center_x = motion_idx * (n_mps_to_show + 1) + (n_mps_to_show - 1) / 2

            if motion_idx % 2 == 0:
                # Even index: place higher
                y_position = y_max + y_range * 0.08
                va = 'bottom'
            else:
                # Odd index: place lower
                y_position = y_max + y_range * 0.02
                va = 'bottom'

            ax.text(center_x, y_position, motion_label,
                    ha='center', va=va, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color_map[motion_id],
                              alpha=0.3, edgecolor='black', linewidth=1))
        ax.set_ylabel('Average Weight ± Std', fontsize=13, fontweight='bold')
        joint_coord_name = CHANNEL_NAMES[joint_idx] if 'CHANNEL_NAMES' in globals() else f'Channel {joint_idx}'
        ax.set_title(f'Weight Distribution of First 5 MPs - {joint_coord_name}',
                     fontsize=15, fontweight='bold', pad=40)  # Increased pad for movement labels

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[motion_id], alpha=0.75,
                                 edgecolor='black', label=motion_labels[i])
                           for i, motion_id in enumerate(unique_motions)]
        ax.legend(handles=legend_elements,
                  loc='center left',  # Position at center left
                  bbox_to_anchor=(1.02, 0.5),  # Place outside plot area
                  fontsize=10, framealpha=0.9,
                  title='Movements', title_fontsize=11,
                  borderaxespad=0)
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
        ax.set_axisbelow(True)  # Grid behind bars
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax.set_ylim(y_min, y_max + y_range * 0.15)
        plt.tight_layout()
        filename = f'{save_dir}/channel_{joint_coord_name.replace(" ", "_")}_weights.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Saved: {filename}")
# def weights_barplot_across_channels(weights, motion_ids, motion_names_dict=None, save_dir='./plots'):
#     """
#     Plot weight distributions across channels.
#
#     For each channel, creates one figure showing the first 5 MP weights
#     for all movements. Each movement is represented by a distinct color.
#
#     Parameters:
#     -----------
#     weights : ndarray, shape (num_segments, num_joints_coord, num_MPs)
#         Weight array
#     motion_ids : ndarray, shape (num_segments,)
#         Motion identifier for each segment
#     motion_names_dict : dict, optional
#         Mapping from motion_id to readable names
#     save_dir : str
#         Directory to save plots
#     """
#     from pathlib import Path
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     Path(save_dir).mkdir(parents=True, exist_ok=True)
#
#     num_segments, num_joints_coord, num_MPs = weights.shape
#     unique_motions = np.unique(motion_ids)
#     n_motions = len(unique_motions)
#
#     # Create motion labels
#     if motion_names_dict:
#         motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
#     else:
#         motion_labels = [f'Motion {m}' for m in unique_motions]
#
#     # Generate distinct colors for each movement
#     colors = plt.cm.tab10(np.linspace(0, 1, n_motions))
#
#     # Number of MPs to display
#     n_mps_to_show = min(5, num_MPs)
#
#     # Iterate over each channel
#     for joint_idx in range(num_joints_coord):
#         fig, ax = plt.subplots(figsize=(14, 7))
#
#         # Calculate bar width (leave some space between MP groups)
#         bar_width = 0.75 / n_motions  # Total width per MP is 0.75
#
#         # For each movement, plot its 5 MP weights
#         for motion_idx, motion_id in enumerate(unique_motions):
#             # Get all segments for this motion
#             mask = motion_ids == motion_id
#             motion_weights = weights[mask, joint_idx, :n_mps_to_show]  # shape: (n_segments, 5)
#
#             # Calculate statistics across segments
#             avg_weights = motion_weights.mean(axis=0)  # shape: (5,)
#             std_weights = motion_weights.std(axis=0)  # shape: (5,)
#
#             # Calculate x positions for this movement's bars
#             # Center the bars for each MP group
#             offset = (motion_idx - (n_motions - 1) / 2) * bar_width
#             x_positions = np.arange(n_mps_to_show) + offset
#
#             # Plot bars with error bars
#             ax.bar(x_positions, avg_weights, width=bar_width,
#                    yerr=std_weights, capsize=4,
#                    color=colors[motion_idx], alpha=0.75,
#                    label=motion_labels[motion_idx],
#                    edgecolor='black', linewidth=0.5)
#
#         # Customize x-axis
#         ax.set_xticks(np.arange(n_mps_to_show))
#         ax.set_xticklabels([f'MP {i + 1}' for i in range(n_mps_to_show)], fontsize=11)
#         ax.set_xlabel('Movement Primitive', fontsize=13, fontweight='bold')
#
#         # Customize y-axis
#         ax.set_ylabel('Average Weight ± Std', fontsize=13, fontweight='bold')
#
#         # Title
#         joint_coord_name = CHANNEL_NAMES[joint_idx] if 'CHANNEL_NAMES' in globals() else f'Channel {joint_idx}'
#         ax.set_title(f'Weight Distribution of First 5 MPs - {joint_coord_name}',
#                      fontsize=15, fontweight='bold', pad=20)
#
#         # Add legend with better formatting
#         ax.legend(loc='upper right', fontsize=10, framealpha=0.9,
#                   title='Movements', title_fontsize=11)
#
#         # Add grid for better readability
#         ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
#         ax.set_axisbelow(True)  # Grid behind bars
#
#         # Add horizontal line at y=0 for reference
#         ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
#
#         # Adjust layout
#         plt.tight_layout()
#
#         # Save figure
#         filename = f'{save_dir}/channel_{joint_coord_name.replace(" ", "_")}_weights.png'
#         plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
#         plt.close()
#
#         print(f"Saved: {filename}")

# def weights_barplot_across_channels(weights, motion_ids, motion_names_dict=None,save_dir='./plots'):
#     Path(save_dir).mkdir(parents=True, exist_ok=True)
#
#     num_segments, num_joints_coord, num_MPs = weights.shape
#
#     unique_motions = np.unique(motion_ids)
#
#     if motion_names_dict:
#         motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
#     else:
#         motion_labels = [f'Motion {m}' for m in unique_motions]
#
#     # for each channel (iterate over num_joints_coord) we have one plot show first 5 MPs weight bars with std across all subject for all movments
#     # (each moveent with one color in one plot  with legend table)
#
#     for joint_idx in range(num_joints_coord):  # Show first 5 MPs
#         n_motions = len(unique_motions)
#
#         for i, motion_id in enumerate(unique_motions):
#             mask = motion_ids == motion_id
#             motion_weights = weights[mask, joint_idx, :5]  # [n_segments, first 5 mp]
#
#             # Average across segments of this motion for each of 5 MPs
#             avg_weights = motion_weights.mean(axis=0)  # [num_mps]
#             std_weights = motion_weights.std(axis=0)
#
#             plt.figure(figsize=(10, 6))
#             joint_coord_name = CHANNEL_NAMES[joint_idx]
#             plt.bar(
#                     yerr=std_weights, width=0.5, capsize=3,
#                     color='steelblue', alpha=0.7)
#             plt.ylabel(f'Average Weight ', fontsize=10)
#             plt.xticks(rotation=45, ha='right', fontsize=6)  # 'ha' for horizontal alignment
#             plt.xlabel('')
#             plt.title(f'First 5 MP Weights distribution for {joint_coord_name} channel across different movements',
#                       fontsize=12, y=1.02)
#             plt.tight_layout()
#             plt.savefig(f'{save_dir}/channel_{joint_coord_name}_weights.png', dpi=150, bbox_inches='tight')
#             plt.close()


#
#
# # ============================================================================
# # ANALYSIS 2: Variance of features (one coordinate) among joints per movement
# # ============================================================================
#
# def analyze_coordinate_variance(weights, motion_ids, motion_names_dict=None,
#                                 coord_idx=0, save_dir='./plots'):
#     """
#     Analyze variance of one coordinate (default: Z) among all joints for each movement.
#
#     Args:
#         weights: [num_segments, num_joints, num_coords, num_MPs]
#         motion_ids: [num_segments] array of motion IDs
#         coord_idx: 0=X, 1=Y, 2=Z
#     """
#     Path(save_dir).mkdir(parents=True, exist_ok=True)
#
#     num_segments, num_joints, num_coords, num_MPs = weights.shape
#     coord_name = COORD_NAMES[coord_idx]
#
#     unique_motions = np.unique(motion_ids)
#
#     # Get motion names
#     if motion_names_dict:
#         motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
#     else:
#         motion_labels = [f'Motion {m}' for m in unique_motions]
#
#     # Extract weights for the specified coordinate
#     coord_weights = weights[:, :, coord_idx, :]  # [num_segments, num_joints, num_MPs]
#
#     results = {}
#
#     for motion_id, motion_label in zip(unique_motions, motion_labels):
#         mask = motion_ids == motion_id
#         motion_weights = coord_weights[mask]  # [n_segs, num_joints, num_MPs]
#
#         # For each MP, calculate variance across joints
#         variances_per_mp = []
#         for mp_idx in range(num_MPs):
#             mp_weights = motion_weights[:, :, mp_idx]  # [n_segs, num_joints]
#             # Variance across joints for each segment, then average
#             var_across_joints = np.var(mp_weights, axis=1).mean()
#             variances_per_mp.append(var_across_joints)
#
#         results[motion_id] = {
#             'label': motion_label,
#             'variances': np.array(variances_per_mp),
#             'mean_variance': np.mean(variances_per_mp),
#             'std_variance': np.std(variances_per_mp)
#         }
#
#     # Plot variance across MPs for each movement
#     fig, ax = plt.subplots(figsize=(14, 7))
#
#     x = np.arange(num_MPs)
#     width = 0.8 / len(unique_motions)
#
#     colors = plt.cm.Set3(np.linspace(0, 1, len(unique_motions)))
#
#     for i, motion_id in enumerate(unique_motions):
#         variances = results[motion_id]['variances']
#         label = results[motion_id]['label']
#         offset = (i - len(unique_motions) / 2) * width + width / 2
#         ax.bar(x + offset, variances, width,
#                label=label, alpha=0.8, color=colors[i])
#
#     ax.set_xlabel('Movement Primitive', fontsize=12, weight='bold')
#     ax.set_ylabel(f'Variance of {coord_name}-coordinate Across Joints', fontsize=12, weight='bold')
#     ax.set_title(f'Variance of {coord_name}-coordinate Among All Joints per Movement',
#                  fontsize=14, weight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels([f'MP{i + 1}' for i in range(num_MPs)], rotation=45, ha='right')
#     ax.legend(loc='upper right', fontsize=10)
#     ax.grid(axis='y', alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/variance_{coord_name}_across_joints.png', dpi=150, bbox_inches='tight')
#     plt.close()
#
#     # Print summary statistics
#     print(f"\n{'=' * 60}")
#     print(f"VARIANCE ANALYSIS: {coord_name}-coordinate across joints")
#     print(f"{'=' * 60}")
#     for motion_id in unique_motions:
#         stats_data = results[motion_id]
#         print(f"\n{stats_data['label']}:")
#         print(f"  Mean variance: {stats_data['mean_variance']:.6f}")
#         print(f"  Std variance:  {stats_data['std_variance']:.6f}")
#
#     print(f"\n✓ Saved variance analysis plot to {save_dir}")
#
#     return results
#
#
# # ============================================================================
# # ANALYSIS 3: Average Z-variance for each joint & significance test
# # ============================================================================
#
# def analyze_joint_coord_variance(weights, motion_ids, motion_names_dict=None, coord_idx=0,save_dir='./plots'):
#     """
#     Calculate average Z-coordinate variance for each joint across MPs.
#     Test statistical significance using one-way ANOVA.
#
#     Args:
#         weights: [num_segments, num_joints, num_coords, num_MPs]
#         motion_ids: [num_segments] array of motion IDs
#     """
#     Path(save_dir).mkdir(parents=True, exist_ok=True)
#
#     num_segments, num_joints, num_coords, num_MPs = weights.shape
#
#     # Extract Z-coordinate weights: [num_segments, num_joints, num_MPs]
#     z_weights = weights[:, :, coord_idx, :]
#
#     # Calculate variance of each joint across MPs (for each segment)
#     joint_variances = []
#     for joint_idx in range(num_joints):
#         joint_z = z_weights[:, joint_idx, :]  # [num_segments, num_MPs]
#         # Variance across MPs for each segment
#         var_per_segment = np.var(joint_z, axis=1)  # [num_segments]
#         joint_variances.append(var_per_segment)
#
#     joint_variances = np.array(joint_variances)  # [num_joints, num_segments]
#
#     # Average variance for each joint
#     avg_joint_variances = joint_variances.mean(axis=1)  # [num_joints]
#     std_joint_variances = joint_variances.std(axis=1)
#
#     # Statistical significance test: Are variances significantly different across joints?
#     # Use one-way ANOVA
#     f_stat, p_value = stats.f_oneway(*[joint_variances[i] for i in range(num_joints)])
#
#     # Create figure with 3 subplots
#     fig = plt.figure(figsize=(18, 6))
#
#     # Subplot 1: Bar plot of average variance per joint
#     ax1 = plt.subplot(1, 3, 1)
#     colors = plt.cm.viridis(np.linspace(0, 1, num_joints))
#     bars = ax1.bar(range(num_joints), avg_joint_variances,
#                    yerr=std_joint_variances,
#                    color=colors, alpha=0.7, capsize=5)
#     ax1.set_xticks(range(num_joints))
#     ax1.set_xticklabels(CHANNEL_NAMES, rotation=45, ha='right', fontsize=9)
#     ax1.set_ylabel('Average Z-coordinate Variance', fontsize=11, weight='bold')
#     ax1.set_title('Average Z-Variance Among MPs\nfor Each Joint', fontsize=12, weight='bold')
#     ax1.grid(axis='y', alpha=0.3)
#
#     # Add significance annotation
#     sig_text = f'ANOVA: F={f_stat:.2f}, p={p_value:.2e}\n' + \
#                ('**SIGNIFICANT**' if p_value < 0.05 else 'Not Significant')
#     ax1.text(0.98, 0.98, sig_text,
#              transform=ax1.transAxes, ha='right', va='top',
#              fontsize=10, weight='bold',
#              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
#
#     # Subplot 2: Box plot showing distribution of variances
#     ax2 = plt.subplot(1, 3, 2)
#     bp = ax2.boxplot([joint_variances[i] for i in range(num_joints)],
#                      labels=JOINT_NAMES, patch_artist=True, showfliers=False)
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.6)
#     ax2.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=9)
#     ax2.set_ylabel(f'{COORD_NAMES[coord_idx]}-coordinate Variance Distribution', fontsize=11, weight='bold')
#     ax2.set_title(f'Distribution of {COORD_NAMES[coord_idx]}-Variance\nfor Each Joint', fontsize=12, weight='bold')
#     ax2.grid(axis='y', alpha=0.3)
#
#     # Subplot 3: Heatmap of variance per joint per motion
#     ax3 = plt.subplot(1, 3, 3)
#     unique_motions = np.unique(motion_ids)
#     variance_matrix = np.zeros((len(unique_motions), num_joints))
#
#     for i, motion_id in enumerate(unique_motions):
#         mask = motion_ids == motion_id
#         motion_z_weights = z_weights[mask]  # [n_segs, num_joints, num_MPs]
#         for j in range(num_joints):
#             joint_z = motion_z_weights[:, j, :]
#             variance_matrix[i, j] = np.var(joint_z, axis=1).mean()
#
#     im = ax3.imshow(variance_matrix, aspect='auto', cmap='YlOrRd')
#     ax3.set_xticks(range(num_joints))
#     ax3.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=9)
#     ax3.set_yticks(range(len(unique_motions)))
#
#     if motion_names_dict:
#         motion_labels = [motion_names_dict.get(m, f'Motion {m}')[:15]
#                          for m in unique_motions]
#     else:
#         motion_labels = [f'Motion {m}' for m in unique_motions]
#     ax3.set_yticklabels(motion_labels, fontsize=9)
#     ax3.set_title(f'{COORD_NAMES[coord_idx]}-Variance Heatmap\n(Motion × Joint)', fontsize=12, weight='bold')
#
#     # Add colorbar
#     cbar = plt.colorbar(im, ax=ax3)
#     cbar.set_label('Variance', fontsize=10)
#
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/joint_{COORD_NAMES[coord_idx]}_variance_analysis.png', dpi=150, bbox_inches='tight')
#     plt.close()
#
#     # Print detailed results
#     print(f"\n{'=' * 60}")
#     print(f"Z-VARIANCE ANALYSIS FOR EACH JOINT")
#     print(f"{'=' * 60}")
#     print(f"\n{'Joint':<15} {'Avg Variance':<15} {'Std Dev':<15}")
#     print(f"{'-' * 45}")
#     for i in range(num_joints):
#         print(f"{JOINT_NAMES[i]:<15} {avg_joint_variances[i]:<15.6f} {std_joint_variances[i]:<15.6f}")
#
#     print(f"\n{'=' * 60}")
#     print(f"STATISTICAL SIGNIFICANCE TEST (One-Way ANOVA)")
#     print(f"{'=' * 60}")
#     print(f"F-statistic: {f_stat:.4f}")
#     print(f"P-value:     {p_value:.2e}")
#     print(f"Result:      {'**SIGNIFICANT** (p < 0.05)' if p_value < 0.05 else 'NOT SIGNIFICANT (p >= 0.05)'}")
#     print(f"\nInterpretation: The Z-coordinate variances across joints are")
#     print(f"                {'statistically different' if p_value < 0.05 else 'not statistically different'}")
#
#     print(f"\n✓ Saved joint Z-variance analysis to {save_dir}")
#
#     return avg_joint_variances, std_joint_variances, f_stat, p_value


def vis_median_weights_movements(weights, motion_ids, motion_names_dict=None,save_dir='./plots'):

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_segments, num_joints_coord, num_MPs = weights.shape

    unique_motions = np.unique(motion_ids)

    if motion_names_dict:
        motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
    else:
        motion_labels = [f'Motion {m}' for m in unique_motions]

    # For each MP, show comparison across joints and movements
    for mp_idx in range(min(num_MPs, 10)):  # Show first 10 MPs
        n_motions = len(unique_motions)

        for i, motion_id in enumerate(unique_motions):
            mask = motion_ids == motion_id
            motion_weights = weights[mask, :, mp_idx]  # [n_segments, num_joints_coord]

            med_weights = np.median(motion_weights, axis=0)

            plt.figure(figsize=(10, 6))
            plt.bar(CHANNEL_NAMES, med_weights,
                   width=0.5 ,  capsize=3,
                   color='steelblue', alpha=0.7)
            plt.ylabel(f'Median Weight ',fontsize=10)
            plt.xticks(rotation=45, ha='right',fontsize=6)  # 'ha' for horizontal alignment
            plt.xlabel('Channels')
            plt.title(f'MP {mp_idx + 1}  Weights for {motion_labels[i]}\n({mask.sum()} segments)',
                      fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/mp_{mp_idx + 1:02d}_{motion_id}_weights.png', dpi=150, bbox_inches='tight')
            plt.close()

    print(f"✓ Saved {min(num_MPs, 10)} MP comparison plots to {save_dir}")


def extract_and_save_avg_weights_for_motions(weights, motion_ids, save_dir, motion_names_dict=None):
    """
    Extract averaged weights for each motion type and save them

    Args:
        weights: numpy array of shape [num_segments, num_joints_coord, num_MPs]
        motion_ids: array of motion IDs for each segment
        save_dir: directory to save averaged weights
        motion_names_dict: optional mapping of motion_id to motion_name

    Returns:
        avg_weights_dict: dictionary mapping motion_id to averaged weights
    """
    from pathlib import Path

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_segments, num_joints_coord, num_MPs = weights.shape
    unique_motions = np.unique(motion_ids)

    avg_weights_dict = {}

    for motion_id in unique_motions:
        # Get all segments for this motion
        mask = motion_ids == motion_id
        motion_weights = weights[mask]  # [n_segments_this_motion, num_joints_coord, num_MPs]

        # Average across segments
        avg_weights = motion_weights.mean(axis=0)  # [num_joints_coord, num_MPs]
        std_weights = motion_weights.std(axis=0)  # [num_joints_coord, num_MPs]

        motion_name = motion_names_dict.get(motion_id,
                                            f'Motion_{motion_id}') if motion_names_dict else f'Motion_{motion_id}'

        # Save averaged weights
        avg_weights_dict[motion_id] = {
            'mean': avg_weights,
            'std': std_weights,
            'n_segments': mask.sum(),
            'motion_name': motion_name
        }

        # Save to file
        np.savez(
            f"{save_dir}/avg_weights_{motion_name}.npz",
            mean_weights=avg_weights,
            std_weights=std_weights,
            n_segments=mask.sum(),
            motion_id=motion_id,
            motion_name=motion_name
        )

        print(f"Motion {motion_id} ({motion_name}): averaged {mask.sum()} segments")

    # # Save all averaged weights together
    # np.savez(
    #     f"{save_dir}/all_avg_weights.npz",
    #     **{f"motion_{mid}": data['mean'] for mid, data in avg_weights_dict.items()},
    #     motion_ids=unique_motions,
    #     motion_names=[avg_weights_dict[mid]['motion_name'] for mid in unique_motions]
    # )

    return avg_weights_dict


def reconstruct_from_weights(weights, MPs, segment_length, kernel_params, resampling_matrix=None):
    """
    Returns:
        reconstructed_segment: numpy array of shape [num_joints, segment_length]
    """
    num_t_points = kernel_params['num_t_points']

    # Compute weighted sum of MPs
    # weights: [num_joints, num_MPs]
    # MPs: [num_MPs, num_t_points]
    # result: [num_joints, num_t_points]
    weighted_mps = np.dot(weights, MPs)

    # Resample to desired segment length
    if segment_length == num_t_points:
        # No resampling needed
        reconstructed = weighted_mps
    else:
        # Need to resample
        if resampling_matrix is None:
            # Compute resampling matrix
            kernel_var = kernel_params['kernel_var']
            kernel_width = kernel_params['kernel_width']
            invK = kernel_params['invK']

            # Create kernel matrix for resampling
            x_new = np.arange(segment_length) * (num_t_points / segment_length)
            x_old = np.arange(num_t_points)

            # RBF kernel
            K_cross = kernel_var * np.exp(-0.5 * (np.subtract.outer(x_new, x_old)) ** 2 / (kernel_width ** 2))

            # Resampling matrix
            resampling_matrix = np.dot(K_cross, invK)

        # Apply resampling: [num_joints, segment_length]
        reconstructed = np.dot(weighted_mps, resampling_matrix.T)

    return reconstructed


def reconstruct_segments_with_avg_weights(model_path, avg_weights_dict, motion_id, segment_lengths):
    """
    Reconstruct multiple segments using averaged weights for a specific motion

    Args:
        model_path: path to saved model
        avg_weights_dict: dictionary from extract_and_save_avg_weights_for_motions
        motion_id: which motion's averaged weights to use
        segment_lengths: list of segment lengths to generate

    Returns:
        reconstructed_segments: list of numpy arrays
    """
    # Load model
    model = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract necessary components
    MPs = model['MPs']
    kernel_params = {
        'kernel_var': model['kernel_var'],
        'kernel_width': model['kernel_width'],
        'num_t_points': model['num_t_points'],
        'K': model['K'],
        'invK': model['invK']
    }

    # Get averaged weights for this motion
    avg_weights = avg_weights_dict[motion_id]['mean']

    # Get resampling matrices if available
    # resampling_matrices = {int(k): v for k, v in model.get('resampling_matrix', {}).items()}
    resampling_matrices = model['resampling_matrix']

    # Reconstruct segments
    reconstructed_segments = []
    for seg_len in segment_lengths:
        resampling_mat = resampling_matrices.get(seg_len, None)

        reconstructed = reconstruct_from_weights(
            weights=avg_weights,
            MPs=MPs,
            segment_length=seg_len,
            kernel_params=kernel_params,
            resampling_matrix=resampling_mat
        )

        reconstructed_segments.append(reconstructed)

    return reconstructed_segments


def main():
    parser = argparse.ArgumentParser(
        description='Analyze TMP model weights across movements and joints',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR,
                        help=f'Directory containing trained models (default: {DEFAULT_MODEL_DIR})')
    parser.add_argument('--bvh-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Directory containing BVH files (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--num-mps', type=int, default=20,
                        help='Number of MPs in model (default: 20)')

    args = parser.parse_args()

    # Load motion mapping
    motion_id_to_name = load_motion_mapping(DEFAULT_MOTION_MAPPING)

    model_subdir = os.path.join(DEFAULT_MODEL_DIR, f"new_seg_pymotion_position_mp_model_5_phase_two")
    model_name = "mp_model_5_PC_tpoints_30"

    model_path = os.path.join(model_subdir,model_name)

    # extract weights form the model
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    print(model_data.keys())
    weights = model_data['weights']

    args.bvh_dir = DEFAULT_DATA_DIR
    folder_path = "../../data/pymotion_position_csv_files"
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(folder_path=folder_path,
                                                                             data_type="position",
                                                                             filtering=False)

    output_dir = os.path.join(model_subdir,"weights_analysis")
    # compare_weights_across_movements(weights, segment_motion_ids,motion_id_to_name,
    #                                  output_dir)
    save_dir = os.path.join(output_dir, "channels_visualization")
    weights_barplot_across_channels(weights, segment_motion_ids,motion_id_to_name,
                                    save_dir = save_dir)
    output_dir = os.path.join(output_dir, "median_weights")
    vis_median_weights_movements(weights, segment_motion_ids,motion_id_to_name,
                                     output_dir)

    output_dir = os.path.join(model_subdir, "averaged_weights")
    avg_weights_dict = extract_and_save_avg_weights_for_motions(
        weights=weights,
        motion_ids=segment_motion_ids,
        save_dir=output_dir,
        motion_names_dict=motion_id_to_name
    )

# build reconstruction array
    # motion_to_reconstruct = 7  # e.g., 'walking'
    # desired_lengths = [200, 60]  # Different segment lengths
    # reconstructed = reconstruct_segments_with_avg_weights(
    #     model_path=model_path,
    #     avg_weights_dict=avg_weights_dict,
    #     motion_id=motion_to_reconstruct,
    #     segment_lengths=desired_lengths
    # )
    # print(reconstructed[0].shape)
    # print(reconstructed[1].shape)
    # output = os.path.join(model_subdir, f"reconstructed_segment_motion_{motion_to_reconstruct}.npy")
    # np.save(output, reconstructed[0])
    # for i, seg in enumerate(reconstructed):
    #     plt.figure(figsize=(12, 6))
    #     for joint_idx in range(min(5, seg.shape[0])):
    #         plt.plot(seg[joint_idx], label=f'Joint {joint_idx}')
    #     plt.title(f'Reconstructed segment (length={desired_lengths[i]})')
    #     plt.legend()
    #     plt.show()

if __name__ == "__main__":
    main()