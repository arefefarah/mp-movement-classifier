#!/usr/bin/env python3
"""
Analyze and visualize TMP model weights across movements and joints.

This script :
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

DEFAULT_MODEL_DIR = "../../results/tmp_configs"
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

    For each channel, creates one figure showing the first 4 MP weights
    grouped by movement. All 4 MPs of one movement share the same color.
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
    n_mps_to_show = min(2, num_MPs)

    # Iterate over each channel
    for joint_idx in range(num_joints_coord):
        # Increase figure size to accommodate legend outside
        fig, ax = plt.subplots(figsize=(18, 8))

        # Calculate positions and data for all movements
        all_avg_weights = []
        all_std_weights = []
        all_median_weights = []
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
            median_weights = np.median(motion_weights, axis=0)

            all_avg_weights.extend(avg_weights)
            all_std_weights.extend(std_weights)
            all_median_weights.extend(median_weights)

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
                      yerr=all_std_weights,
                      capsize=4,
                      color=bar_colors, alpha=0.75,
                      edgecolor='black', linewidth=0.5,
                      width=0.8)
        #plot median
        # bars = ax.bar(x_positions, all_median_weights,
        #               capsize=4,
        #               color=bar_colors, alpha=0.75,
        #               edgecolor='black', linewidth=0.5,
        #               width=0.8)

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
        ax.set_title(f'Weight Distribution of First {n_mps_to_show} MPs - {joint_coord_name}',
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


def mean_weights_barplot_mp1_subset_channels(
        weights,
        motion_ids,
        motion_names_dict=None,
        channel_names=("LWrist_Zpos", "RWrist_Zpos",  "LAnkle_Zpos","LAnkle_Ypos","Neck_Zpos","RElbow_Ypos",
                       "LElbow_Ypos"),
        mp_idx=0,
        save_dir='./plots',
):
    """
    Bar plot of one MP coefficient (default MP1 = ``mp_idx=0``) across a
    chosen *subset* of channels, grouped by movement.

    Parameters
    ----------
    weights : ndarray, shape (num_segments, num_channels, num_MPs)
    motion_ids : ndarray, shape (num_segments,)
    motion_names_dict : dict[int, str], optional
        Mapping motion_id -> readable name.
    channel_names : sequence of str
        Channels to plot, matched (case-insensitive) against
        ``CHANNEL_NAMES``. Accepts both ``"LWrist_z"`` and
        ``"LWrist_Zpos"`` styles.
    mp_idx : int
        Zero-based MP index (0 = MP1).
    save_dir : str
        Directory in which to save the figure.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ----- Resolve channel names -> indices (case-insensitive, suffix-tolerant) -----
    def _resolve(name: str) -> int:
        # Accept "LWrist_z" or "LWrist_Zpos" (any case).
        n = name.strip().lower()
        for idx, cname in enumerate(CHANNEL_NAMES):
            cl = cname.lower()
            if cl == n or cl == n + "pos":
                return idx
        raise ValueError(
            f"Channel '{name}' not found in CHANNEL_NAMES "
            f"(expected one of e.g. 'LWrist_Zpos')."
        )

    channel_idxs = [_resolve(c) for c in channel_names]
    channel_labels = [CHANNEL_NAMES[i] for i in channel_idxs]
    n_channels = len(channel_idxs)

    if mp_idx < 0 or mp_idx >= weights.shape[2]:
        raise ValueError(f"mp_idx={mp_idx} out of range [0, {weights.shape[2]}).")

    # ----- Motions (sorted) + readable labels -----
    unique_motions = np.unique(motion_ids)
    n_motions = len(unique_motions)
    if motion_names_dict:
        motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
    else:
        motion_labels = [f'Motion {m}' for m in unique_motions]

    # ----- Color map (same palette as weights_barplot_across_channels) -----
    fixed_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    ]
    color_map = {m: fixed_colors[i % len(fixed_colors)]
                 for i, m in enumerate(unique_motions)}

    # Width is fixed (paper column constraint); height is back to standard
    # since the movement labels above the bars have been removed.
    fig, ax = plt.subplots(figsize=(18, 8))

    # Shorter, paper-friendly form for the legend (e.g. "LWrist_Zpos" -> "LWrist_z").
    def _short(name: str) -> str:
        n = name
        for suf, axis in (("_Xpos", "_x"), ("_Ypos", "_y"), ("_Zpos", "_z")):
            if n.endswith(suf):
                return n[:-len(suf)] + axis
        return n

    short_channel_labels = [_short(c) for c in channel_labels]

    avg_vals, std_vals = [], []
    x_positions, bar_colors, x_labels = [], [], []

    current_x = 0
    for m in unique_motions:
        mask = motion_ids == m
        motion_color = color_map[m]
        for ch_pos, ch_idx in enumerate(channel_idxs):
            vals = weights[mask, ch_idx, mp_idx]
            avg_vals.append(vals.mean())
            std_vals.append(vals.std())
            x_positions.append(current_x)
            bar_colors.append(motion_color)
            # X-tick is just the channel *index* (1-based) within the chosen
            # subset; the index ↔ name mapping lives in the legend.
            x_labels.append(str(ch_pos + 1))
            current_x += 1
        current_x += 1  # gap between movement groups

    ax.bar(x_positions, avg_vals,
           yerr=std_vals, capsize=4,
           color=bar_colors, alpha=0.75,
           edgecolor='black', linewidth=0.5,
           width=0.95)  # near-touching within a group → fewer wasted gaps

    # ----- X-axis: short index under each bar (no name crowding) -----
    # All font sizes in this figure are bumped ~1.5× so it scales down
    # cleanly into a paper column without becoming illegible.
    ax.set_xticks(x_positions)
    # X-tick font matches the y-tick font for visual consistency.
    ax.set_xticklabels(x_labels, fontsize=14, rotation=0, ha='center')
    ax.set_xlabel('Channel index (grouped by movement)',
                  fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    # Trim the empty space at the left/right of the plot so the bars fill
    # the full axes width.
    ax.set_xlim(x_positions[0] - 0.6, x_positions[-1] + 0.6)
    ax.margins(x=0)

    # ----- Vertical separators between movement groups -----
    group_size = n_channels + 1  # bars + gap
    for k in range(1, n_motions):
        ax.axvline(x=k * group_size - 1, color='gray',
                   linestyle='--', linewidth=1.5, alpha=0.5)

    # Movement labels above each group are intentionally omitted: the
    # right-side movement legend already conveys the movement ↔ color mapping
    # and adding boxes here either overlaps neighbours or steals plot height.
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    ax.set_ylabel(f'Average Weight ± Std (MP{mp_idx + 1})',
                  fontsize=20, fontweight='bold')
    # Compact y-tick labels: show "1", "5", "10" with a single "×10⁻³"
    # multiplier at the top of the axis instead of "0.001", "0.005",
    # "0.010" on every tick. Frees up horizontal space → bars look bigger.
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0),
                        useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(13)
    # No movement-label strip above bars, so the standard small title pad
    # is enough.
    ax.set_title(
        f'MP{mp_idx + 1} Weight Distribution: '
        f'{len(channel_idxs)} channels × {n_motions} movements',
        fontsize=22, fontweight='bold', pad=20,
    )

    # ----- Two legends on the right: channel index ↔ name, and movement ↔ color.
    # Use invisible proxies for the index legend so we get a clean text-only
    # mapping (no color swatches — index isn't a color, just a position).
    from matplotlib.lines import Line2D
    index_handles = [
        Line2D([0], [0], marker='', linestyle='',
               label=f'{i + 1}: {short_channel_labels[i]}')
        for i in range(n_channels)
    ]
    motion_handles = [
        Patch(facecolor=color_map[m], alpha=0.75,
              edgecolor='black', label=motion_labels[i])
        for i, m in enumerate(unique_motions)
    ]

    # Channel-index legend: 3 columns (≈ 3 entries per column for 7 channels).
    leg_idx = ax.legend(
        handles=index_handles,
        loc='upper left', bbox_to_anchor=(0.0, -0.12),
        ncol=3,
        fontsize=15, framealpha=0.9,
        title='Channel index', title_fontsize=16,
        borderaxespad=0, handlelength=0, handletextpad=0,
        columnspacing=1.2,
    )
    ax.add_artist(leg_idx)

    # Movement-color legend: 4×4 grid for 16 movements.
    ax.legend(
        handles=motion_handles,
        loc='upper right', bbox_to_anchor=(1.0, -0.12),
        ncol=4,
        fontsize=15, framealpha=0.9,
        title='Movements', title_fontsize=16,
        borderaxespad=0, columnspacing=1.5, handletextpad=0.6,
    )

    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    # No label strip above the bars now; small headroom is enough.
    ax.set_ylim(y_min, y_max + y_range * 0.1)

    plt.tight_layout()
    fname = (f"{save_dir}/MP{mp_idx + 1}_subset_"
             f"{'_'.join(channel_labels)}.png")
    plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(fname.replace('.png', '.svg'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fname}")
    return fname



def median_weights_barplot_mp1_subset_channels(
        weights,
        motion_ids,
        motion_names_dict=None,
        channel_names=("LWrist_Zpos", "RWrist_Zpos", "Neck_Zpos","RElbow_Ypos",
                       "LElbow_Ypos"),mp_idx=0,save_dir='./plots'):

    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ----- Resolve channel names -> indices (case-insensitive, suffix-tolerant) -----
    def _resolve(name: str) -> int:
        # Accept "LWrist_z" or "LWrist_Zpos" (any case).
        n = name.strip().lower()
        for idx, cname in enumerate(CHANNEL_NAMES):
            cl = cname.lower()
            if cl == n or cl == n + "pos":
                return idx
        raise ValueError(
            f"Channel '{name}' not found in CHANNEL_NAMES "
            f"(expected one of e.g. 'LWrist_Zpos')."
        )

    channel_idxs = [_resolve(c) for c in channel_names]
    channel_labels = [CHANNEL_NAMES[i] for i in channel_idxs]
    n_channels = len(channel_idxs)

    if mp_idx < 0 or mp_idx >= weights.shape[2]:
        raise ValueError(f"mp_idx={mp_idx} out of range [0, {weights.shape[2]}).")

    # ----- Motions (sorted) + readable labels -----
    unique_motions = np.unique(motion_ids)
    n_motions = len(unique_motions)
    if motion_names_dict:
        motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
    else:
        motion_labels = [f'Motion {m}' for m in unique_motions]

    # ----- Color map (same palette as weights_barplot_across_channels) -----
    fixed_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    ]
    color_map = {m: fixed_colors[i % len(fixed_colors)]
                 for i, m in enumerate(unique_motions)}

    # Width is fixed (paper column constraint); height is back to standard
    # since the movement labels above the bars have been removed.
    fig, ax = plt.subplots(figsize=(18, 8))

    # Shorter, paper-friendly form for the legend (e.g. "LWrist_Zpos" -> "LWrist_z").
    def _short(name: str) -> str:
        n = name
        for suf, axis in (("_Xpos", "_x"), ("_Ypos", "_y"), ("_Zpos", "_z")):
            if n.endswith(suf):
                return n[:-len(suf)] + axis
        return n

    short_channel_labels = [_short(c) for c in channel_labels]

    med_vals, std_vals = [], []
    x_positions, bar_colors, x_labels = [], [], []

    current_x = 0
    for m in unique_motions:
        mask = motion_ids == m
        motion_color = color_map[m]
        for ch_pos, ch_idx in enumerate(channel_idxs):
            vals = weights[mask, ch_idx, mp_idx]
            med_vals.append(np.median(vals, axis=0))
            # std_vals.append(vals.std())
            x_positions.append(current_x)
            bar_colors.append(motion_color)
            x_labels.append(str(ch_pos + 1))
            current_x += 1
        current_x += 1  # gap between movement groups

    ax.bar(x_positions, med_vals,
           # yerr=std_vals,
           capsize=4,
           color=bar_colors, alpha=0.75,
           edgecolor='black', linewidth=0.5,
           width=0.95)  # near-touching within a group → fewer wasted gaps

    # ----- X-axis: short index under each bar (no name crowding) -----
    # All font sizes in this figure are bumped ~1.5× so it scales down
    # cleanly into a paper column without becoming illegible.
    ax.set_xticks(x_positions)
    # X-tick font matches the y-tick font for visual consistency.
    ax.set_xticklabels(x_labels, fontsize=14, rotation=0, ha='center')
    ax.set_xlabel('Channel index (grouped by movement)',
                  fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    # Trim the empty space at the left/right of the plot so the bars fill
    # the full axes width.
    ax.set_xlim(x_positions[0] - 0.6, x_positions[-1] + 0.6)
    ax.margins(x=0)

    # ----- Vertical separators between movement groups -----
    group_size = n_channels + 1  # bars + gap
    for k in range(1, n_motions):
        ax.axvline(x=k * group_size - 1, color='gray',
                   linestyle='--', linewidth=1.5, alpha=0.5)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    ax.set_ylabel(f'Average Weight ± Std (MP{mp_idx + 1})',
                  fontsize=20, fontweight='bold')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0),
                        useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(13)
    # No movement-label strip above bars, so the standard small title pad
    # is enough.
    ax.set_title(
        f'Median of MP{mp_idx + 1} Weight Distribution: '
        f'{len(channel_idxs)} channels × {n_motions} movements',
        fontsize=22, fontweight='bold', pad=20,
    )

    # ----- Two legends on the right: channel index ↔ name, and movement ↔ color.
    # Use invisible proxies for the index legend so we get a clean text-only
    # mapping (no color swatches — index isn't a color, just a position).
    from matplotlib.lines import Line2D
    index_handles = [
        Line2D([0], [0], marker='', linestyle='',
               label=f'{i + 1}: {short_channel_labels[i]}')
        for i in range(n_channels)
    ]
    motion_handles = [
        Patch(facecolor=color_map[m], alpha=0.75,
              edgecolor='black', label=motion_labels[i])
        for i, m in enumerate(unique_motions)
    ]
    # Channel-index legend: 3 columns (≈ 3 entries per column for 7 channels).
    leg_idx = ax.legend(
        handles=index_handles,
        loc='upper left', bbox_to_anchor=(0.0, -0.12),
        ncol=3,
        fontsize=15, framealpha=0.9,
        title='Channel index', title_fontsize=16,
        borderaxespad=0, handlelength=0, handletextpad=0,
        columnspacing=1.2,
    )
    ax.add_artist(leg_idx)

    # Movement-color legend: 4×4 grid for 16 movements.
    ax.legend(
        handles=motion_handles,
        loc='upper right', bbox_to_anchor=(1.0, -0.12),
        ncol=4,
        fontsize=15, framealpha=0.9,
        title='Movements', title_fontsize=16,
        borderaxespad=0, columnspacing=1.5, handletextpad=0.6,
    )

    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    # No label strip above the bars now; small headroom is enough.
    ax.set_ylim(y_min, y_max + y_range * 0.1)

    plt.tight_layout()
    fname = (f"{save_dir}/median_MP{mp_idx + 1}_subset_"
             f"{'_'.join(channel_labels)}.png")
    plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(fname.replace('.png', '.svg'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fname}")
    return fname

def vis_median_weights_movements(weights, motion_ids, motion_names_dict=None,save_dir='./plots'):

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_segments, num_joints_coord, num_MPs = weights.shape

    unique_motions = np.unique(motion_ids)

    if motion_names_dict:
        motion_labels = [motion_names_dict.get(m, f'Motion {m}') for m in unique_motions]
    else:
        motion_labels = [f'Motion {m}' for m in unique_motions]

    # For each MP, show comparison across joints and movements
    for mp_idx in range(min(num_MPs, 5)):  # Show first 5 MPs
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

    return avg_weights_dict

#
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


def reconstruct_segments_with_avg_weights(model_path, avg_weights, segment_length):
    """
    Reconstruct multiple segments using averaged weights for a specific motion

    Args:
        model_path: path to saved model
        avg_weights: averaged weights for this motion
        segment_length: segment length to generate

    Returns:
        reconstructed_segment: numpy arrays
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
    avg_weights = avg_weights
    resampling_matrices = model['resampling_matrix']
    resampling_mat = resampling_matrices.get(segment_length, None)

    reconstructed = reconstruct_from_weights(
        weights=avg_weights,
        MPs=MPs,
        segment_length=segment_length,
        kernel_params=kernel_params,
        resampling_matrix=resampling_mat
    )
    return reconstructed


def main():
    num_MPs = 5
    tpoints = 35

    model_subdir = os.path.join("./../../results/tmp_configs", f"new_seg_mp_model_{num_MPs}_phase_three")
    model_path = os.path.join(
        model_subdir,
        f"mp_model_{num_MPs}_PC_tpoints_{tpoints}"
    )

    output_dir = os.path.join(model_subdir, "weights_analysis")
    folder_path = "../../data/pymotion_position_csv_files"

    # Load motion mapping
    motion_id_to_name = load_motion_mapping(DEFAULT_MOTION_MAPPING)
    # extract weights form the model
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    print(model_data.keys())
    weights = model_data['weights']

    motion_ids, processed_segments, segment_motion_ids = process_motion_data(folder_path=folder_path,
                                                                             data_type="position",
                                                                             filtering=False)
    ####
    # Visulaization of weights
    ####

    # barplot of all joints for specific MP across all segments of one specific movment
    # compare_weights_across_movements(weights, segment_motion_ids,motion_id_to_name,
    #                                  output_dir)

    # save_dir = os.path.join(output_dir, "channels_visualization")
    # weights_barplot_across_channels(weights, segment_motion_ids,motion_id_to_name,
    #                                 save_dir = save_dir)

    # MP1 across a curated subset of channels (one figure)
    save_dir_mp1 = os.path.join(output_dir, "mp1_subset_channels")
    mean_weights_barplot_mp1_subset_channels(
        weights, segment_motion_ids, motion_id_to_name,
        channel_names=("Neck_Zpos", "LWrist_Zpos", "RWrist_Zpos","RElbow_Ypos",
                       "LElbow_Ypos"),
        mp_idx=0,
        save_dir=save_dir_mp1,
    )

    median_weights_barplot_mp1_subset_channels(
        weights, segment_motion_ids, motion_id_to_name,
        channel_names=("Neck_Zpos", "LWrist_Zpos", "RWrist_Zpos", "RElbow_Ypos",
                       "LElbow_Ypos"),
        mp_idx=0,
        save_dir=save_dir_mp1,
    )

    # save_dir = os.path.join(output_dir, "median_weights")
    # vis_median_weights_movements(weights, segment_motion_ids,motion_id_to_name,
    #                                  save_dir)

    ####
    # Extract and save weights
    ####
    # output_dir = os.path.join(model_subdir, "averaged_weights")
    # avg_weights_dict = extract_and_save_avg_weights_for_motions(
    #     weights=weights,
    #     motion_ids=segment_motion_ids,
    #     save_dir=output_dir,
    #     motion_names_dict=motion_id_to_name
    # )

if __name__ == "__main__":
    main()