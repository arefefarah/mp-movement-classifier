"""
Filter MMPose Position Data (CSV)
Following Le et al. (2023) methodology - filtering at the position stage

This script applies Butterworth low-pass filtering to 3D position data
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os
import warnings
from pathlib import Path


def apply_butterworth_filter_to_positions(positions, cutoff_freq=6.0, filter_order=4,
                                          sampling_freq=30, axis=0):
    """
    Apply Butterworth filter to position trajectories

    Returns:
        filtered_positions: same shape as input
    """
    nyquist_freq = sampling_freq / 2.0

    # Validate cutoff frequency
    if cutoff_freq >= nyquist_freq:
        print(f"⚠️  Warning: Cutoff ({cutoff_freq} Hz) >= Nyquist ({nyquist_freq} Hz)")
        cutoff_freq = nyquist_freq * 0.8
        print(f"   Adjusting to {cutoff_freq:.1f} Hz")

    # Normalize cutoff frequency
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Design Butterworth filter
    b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)

    # Apply filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered_positions = filtfilt(b, a, positions, axis=axis)

    return filtered_positions


def filter_mmpose_csv(csv_file, output_file=None, cutoff_freq=6.0,
                      filter_order=4, sampling_freq=30,
                      coordinate_columns=None):
    """
    Filter MMPose CSV file containing position data

    Returns:
        DataFrame with filtered positions
    """

    # Read CSV
    df = pd.read_csv(csv_file)

    joints = df['joint_name'].unique()
    df_filtered = df.copy()

    # Filter each coordinate column for each joint
    for joint in joints:
        for col in coordinate_columns:
            original_data = df.loc[df['joint_name'] == joint, col].values
            filtered_data = apply_butterworth_filter_to_positions(
                original_data,
                cutoff_freq=cutoff_freq,
                filter_order=filter_order,
                sampling_freq=sampling_freq,
                axis=0
            )
            df_filtered.loc[df_filtered['joint_name'] == joint, col] = filtered_data



    # # Calculate filtering metrics
    original_positions = df[coordinate_columns].values
    filtered_positions = df_filtered[coordinate_columns].values

    variance_retained = (np.var(filtered_positions) / np.var(original_positions)) * 100
    rmse = np.sqrt(np.mean((original_positions - filtered_positions) ** 2))


    output_file = csv_file.name

    df_filtered.to_csv(f"../../data/MMpose/filtered_csv_files/{output_file}", index=False)

    return df_filtered,variance_retained,rmse


def visualize_filtering_effect(filtered_df,csv_file,
                               joint_to_plot="LKnee", coord_to_plot='x_3d',
                               output_dir='../../data/MMpose/filtered_csv_files/comparison_analysis'):
    """
    Visualize the effect of filtering on a specific joint coordinate

    Args:
        original_df: Original DataFrame
        filtered_df: Filtered DataFrame
        output_dir: Where to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    original_df = pd.read_csv(csv_file)

    original_col = original_df.loc[original_df["joint_name" ]== joint_to_plot,coord_to_plot].values

    filtered_col = filtered_df.loc[filtered_df["joint_name" ]== joint_to_plot,coord_to_plot].values

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    time = np.arange(original_col.shape[0]) / 30.0  # Assuming 30 Hz

    # Plot 1: Time domain
    ax1.plot(time, original_col, 'b-', alpha=0.6, linewidth=1, label='Original')
    ax1.plot(time, filtered_col, 'r-', linewidth=2, label='Filtered')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position')
    ax1.set_title(f'Position Trajectory: {coord_to_plot} for joint {joint_to_plot}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difference (noise removed)
    difference = original_col- filtered_col
    ax2.plot(time, difference, 'g-', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Difference (removed signal)')
    ax2.set_title('High-Frequency Content Removed by Filter')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{csv_file.stem}_{joint_to_plot}_{coord_to_plot}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    path = Path("../../data/MMpose/df_files_3d")
    # csv_file = Path("../../data/MMpose/df_files_3d/subject_12_motion_06.csv")

    filtering_result = []
    for csv_file in path.glob("*.csv"):
        coordinate_columns = ['x_3d', 'y_3d', 'z_3d']
        df_filtered, variance_retained, rmse = filter_mmpose_csv(
            csv_file=csv_file,
            coordinate_columns=coordinate_columns,
            cutoff_freq=6.0,
            filter_order=4,
            sampling_freq=30
        )
        print(f"Filtering completed for {csv_file.stem} ")
        output_dir = '../../data/MMpose/filtered_csv_files/comparison_analysis'
        visualize_filtering_effect(df_filtered,csv_file,
                                       joint_to_plot="LKnee", coord_to_plot='x_3d',output_dir=output_dir)

        res = [csv_file.stem,variance_retained,rmse]
        filtering_result.append(res)

    df = pd.DataFrame(filtering_result,columns=['subject','variance_retained','rmse'])
    df.to_csv(f"../../data/MMpose/filtered_csv_files/comparison_analysis/All_filtering_result.csv",index=False)


if __name__ == "__main__":
    main()