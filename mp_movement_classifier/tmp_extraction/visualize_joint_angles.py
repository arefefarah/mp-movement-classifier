import numpy as np
import re
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import warnings


def parse_bvh_robust(file_path):
    """
    Robust BVH parser that handles various format issues
    """
    with open(file_path, 'r') as file:
        content = file.read()


    # Split into hierarchy and motion sections
    parts = content.split('MOTION')
    if len(parts) < 2:
        print("❌ Invalid BVH format: No MOTION section found")
        return None, None, None, None

    hierarchy = parts[0]
    motion_part = parts[1]

    # Extract joint information
    joints = {}
    channel_index = 0

    # Find all joints and their channels
    joint_pattern = r'(ROOT|JOINT)\s+(\w+)'
    channel_pattern = r'CHANNELS\s+(\d+)\s+(.*)'

    lines = hierarchy.split('\n')
    current_joint = None

    for line in lines:
        line = line.strip()

        # Find joint names
        joint_match = re.search(joint_pattern, line)
        if joint_match:
            current_joint = joint_match.group(2)

        # Find channels
        channel_match = re.search(channel_pattern, line)
        if channel_match and current_joint:
            num_channels = int(channel_match.group(1))
            channels = channel_match.group(2).split()

            joints[current_joint] = {
                'channels': channels,
                'start_index': channel_index
            }
            channel_index += num_channels

    # Extract motion data
    motion_lines = motion_part.strip().split('\n')

    # Get frame info
    frames = 0
    frame_time = 0.0

    for line in motion_lines:
        if line.startswith('Frames:'):
            frames = int(line.split(':')[1].strip())
        elif line.startswith('Frame Time:'):
            frame_time = float(line.split(':')[1].strip())

    if frames == 0:
        print("❌ No frame information found in BVH file")
        return None, None, None, None

    # Extract frame data
    frame_data = []

    for line in motion_lines:
        line = line.strip()
        if line and not line.startswith('Frames') and not line.startswith('Frame Time'):
            try:
                values = [float(x) for x in line.split()]
                frame_data.extend(values)
            except ValueError:
                continue

    # Calculate expected total channels
    total_channels = sum(len(joint['channels']) for joint in joints.values())
    expected_data_points = total_channels * frames

    if len(frame_data) < expected_data_points:
        print(f"⚠️ Warning: Less data than expected. Using available frames.")
        available_frames = len(frame_data) // total_channels
        motion_data = np.array(frame_data[:available_frames * total_channels]).reshape(available_frames, total_channels)
        frames = available_frames
    else:
        motion_data = np.array(frame_data[:expected_data_points]).reshape(frames, total_channels)

    return joints, motion_data, frame_time, frames


def apply_butterworth_smoothing(motion_data, frame_time, cutoff_freq=6.0, filter_order=6):
    """
    Apply sixth-order Butterworth filter to motion capture data

    Based on research findings:
    - Sixth-order zero-lag Butterworth filters are widely recommended
    - 6 Hz cutoff is optimal for general human motion
    - Bidirectional filtering prevents phase distortion

    Args:
        motion_data: numpy array of shape (frames, channels)
        frame_time: time between frames in seconds
        cutoff_freq: cutoff frequency in Hz (default 6.0)
        filter_order: filter order (default 6)

    Returns:
        smoothed_data: filtered motion data
    """

    # Calculate sampling frequency
    sampling_freq = 1.0 / frame_time
    nyquist_freq = sampling_freq / 2.0

    # Validate cutoff frequency
    if cutoff_freq >= nyquist_freq:
        print(f"⚠️ Warning: Cutoff frequency ({cutoff_freq} Hz) is too high for sampling rate ({sampling_freq:.1f} Hz)")
        cutoff_freq = nyquist_freq * 0.8  # Use 80% of Nyquist frequency
        print(f"   Adjusting cutoff to {cutoff_freq:.1f} Hz")

    # Normalize cutoff frequency
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Design Butterworth filter
    b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)

    # Apply filter to each channel
    smoothed_data = np.zeros_like(motion_data)

    # Suppress warnings for small datasets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for channel in range(motion_data.shape[1]):
            # Use filtfilt for zero-phase filtering (bidirectional)
            smoothed_data[:, channel] = filtfilt(b, a, motion_data[:, channel])

    return smoothed_data


def extract_joint_angles_robust(joints, motion_data, joint_name):
    """
    Extract rotation angles for a specific joint with error handling
    """
    if joint_name not in joints:
        available_joints = list(joints.keys())
        print(f"Joint '{joint_name}' not found.")
        print(f"Available joints: {available_joints}")
        return None

    joint_info = joints[joint_name]
    start_idx = joint_info['start_index']
    channels = joint_info['channels']

    angles = {}
    for i, channel in enumerate(channels):
        if 'rotation' in channel.lower():
            if start_idx + i < motion_data.shape[1]:
                angles[channel] = motion_data[:, start_idx + i]
            else:
                print(f"Channel index out of range for {joint_name}.{channel}")

    return angles if angles else None


def calculate_smoothing_metrics(original_data, smoothed_data):
    """
    Calculate metrics to quantify the effect of smoothing
    """
    metrics = {}

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((original_data - smoothed_data) ** 2))
    metrics['RMSE'] = rmse

    # Maximum absolute difference
    max_diff = np.max(np.abs(original_data - smoothed_data))
    metrics['Max_Diff'] = max_diff

    # Correlation coefficient
    correlation = np.corrcoef(original_data.flatten(), smoothed_data.flatten())[0, 1]
    metrics['Correlation'] = correlation

    # Noise reduction (difference in standard deviation)
    original_std = np.std(original_data)
    smoothed_std = np.std(smoothed_data)
    noise_reduction = ((original_std - smoothed_std) / original_std) * 100
    metrics['Noise_Reduction_%'] = noise_reduction

    return metrics


def visualize_joint_angles_with_smoothing(bvh_filename, target_joints=None, apply_smoothing=True,
                                          cutoff_freq=6.0, filter_order=6):
    """
    Robust visualization with optional Butterworth smoothing
    """
    if target_joints is None:
        target_joints = ['LeftWrist', 'RightWrist', 'LeftKnee', 'RightKnee', 'LeftAnkle']

    # Parse BVH file
    file_path = f"../../data/bvh_files/{bvh_filename}.bvh"
    joints, motion_data, frame_time, frames = parse_bvh_robust(file_path)
    print(frame_time)

    if joints is None:
        return None, None, None, None

    # Apply smoothing if requested
    smoothed_data = None
    if apply_smoothing:
        print(f"\nApplying smoothing...")
        smoothed_data = apply_butterworth_smoothing(motion_data, frame_time, cutoff_freq, filter_order)

    # Create time vector
    time_vector = np.arange(frames) * frame_time

    # Filter target joints to only those that exist
    valid_joints = [j for j in target_joints if j in joints]
    invalid_joints = [j for j in target_joints if j not in joints]

    if invalid_joints:
        print(f"\nSkipping non-existent joints: {invalid_joints}")

    if not valid_joints:
        print("No valid joints found to plot")
        return joints, motion_data, frame_time, frames

    # Create plots
    fig, axes = plt.subplots(len(valid_joints), 1, figsize=(16, 5 * len(valid_joints)))
    if len(valid_joints) == 1:
        axes = [axes]

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

    print(f"\n Plotting results:")

    for i, joint_name in enumerate(valid_joints):
        # Extract original angles
        original_angles = extract_joint_angles_robust(joints, motion_data, joint_name)

        # Extract smoothed angles if available
        smoothed_angles = None
        if smoothed_data is not None:
            smoothed_angles = extract_joint_angles_robust(joints, smoothed_data, joint_name)

        if original_angles is None:
            continue

        ax = axes[i] if len(valid_joints) > 1 else axes[0]
        ax.set_title(f'{joint_name} Joint Angles' + (' - Original vs Smoothed' if apply_smoothing else ''),
                     fontsize=16, fontweight='bold')

        # Plot each rotation channel
        for j, (channel, original_data) in enumerate(original_angles.items()):
            if j < len(colors):
                # Plot original data
                ax.plot(time_vector, original_data,
                        color=colors[j],
                        label=f'{channel} (Original)',
                        linewidth=1.5,
                        alpha=0.7,
                        linestyle='-')

                # Plot smoothed data if available
                if smoothed_angles and channel in smoothed_angles:
                    smoothed_channel_data = smoothed_angles[channel]
                    ax.plot(time_vector, smoothed_channel_data,
                            color=colors[j],
                            label=f'{channel} (Smoothed)',
                            linewidth=2.5,
                            alpha=0.9,
                            linestyle='--')

                    # Calculate and display metrics for this channel
                    metrics = calculate_smoothing_metrics(original_data, smoothed_channel_data)
                    print(f"   {joint_name}.{channel}: RMSE={metrics['RMSE']:.3f}°, "
                          f"Correlation={metrics['Correlation']:.3f}, "
                          f"Noise reduction={metrics['Noise_Reduction_%']:.1f}%")

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time_vector[-1])

        # Add statistics
        if original_angles:
            all_original = np.concatenate(list(original_angles.values()))
            stats_text = f'Original - Range: [{np.min(all_original):.1f}, {np.max(all_original):.1f}]°\n'
            stats_text += f'Mean: {np.mean(all_original):.1f}° | Std: {np.std(all_original):.1f}°'

            if smoothed_angles:
                all_smoothed = np.concatenate(list(smoothed_angles.values()))
                stats_text += f'\nSmoothed - Range: [{np.min(all_smoothed):.1f}, {np.max(all_smoothed):.1f}]°\n'
                stats_text += f'Mean: {np.mean(all_smoothed):.1f}° | Std: {np.std(all_smoothed):.1f}°'

                # Overall improvement metrics
                overall_metrics = calculate_smoothing_metrics(all_original, all_smoothed)
                stats_text += f'\nCorrelation: {overall_metrics["Correlation"]:.3f}'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    # Save plot with appropriate filename
    suffix = f"_smoothed_{cutoff_freq}Hz" if apply_smoothing else "_original"
    figures_dir = os.path.join("./../../results", 'joints_visualizations')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"{bvh_filename}_joint_angles{suffix}.png"),dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved as: {figures_dir}")

    plt.close()

    return joints, motion_data, frame_time, frames, smoothed_data


def main():

    bvh_filename = "subject_2_motion_04"
    bvh_file = f"../../data/bvh_files/{bvh_filename}.bvh"

    # Customize these joints based on what you want to analyze
    joints_to_analyze = ['LeftWrist', 'RightWrist', 'LeftKnee', 'RightKnee', 'LeftAnkle']

    print(" BVH Joint Angle Analysis with Butterworth Smoothing")
    print("=" * 80)

    # visualize original data without smoothing
    # result_original = visualize_joint_angles_with_smoothing(
    #     bvh_file,
    #     joints_to_analyze,
    #     apply_smoothing=False
    # )

    # Apply smoothing and visualize
    print("\n Applying Butterworth smoothing and comparing with original data... ...")
    result_smoothed = visualize_joint_angles_with_smoothing(
        bvh_filename,
        joints_to_analyze,
        apply_smoothing=True,
        cutoff_freq=3.0,  # 6 Hz as recommended by research
        filter_order=4
    )

if __name__ == "__main__":
    main()

