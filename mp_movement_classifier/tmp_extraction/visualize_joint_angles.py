import numpy as np
import re
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt,welch
from scipy.signal import find_peaks
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


def apply_butterworth_smoothing(motion_data, cutoff_freq=6.0, filter_order=6):
    """
    Apply sixth-order Butterworth filter to motion capture data

    Based on research findings:
    - Sixth-order zero-lag Butterworth filters are widely recommended
    - 6 Hz cutoff is optimal for general human motion
    - Bidirectional filtering prevents phase distortion

    Args:
        motion_data: numpy array of shape (frames, channels)
        cutoff_freq: cutoff frequency in Hz (default 6.0)
        filter_order: filter order (default 6)

    Returns:
        smoothed_data: filtered motion data
    """

    # Calculate sampling frequency
    sampling_freq = 30
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


def compute_joint_speed(motion_data, joints, frame_time, wrist_joints=['LeftWrist', 'RightWrist'],
                        ankle_joints=['LeftAnkle', 'RightAnkle']):
    """
    Compute speed of specified joints

    Args:
        motion_data: Smoothed motion capture data
        joints: Joint mapping from BVH parser
        frame_time: Time between frames
        wrist_joints: List of wrist joint names
        ankle_joints: List of ankle joint names

    Returns:
        Joint speed array
    """
    # Initialize speed array
    joint_speeds = np.zeros(motion_data.shape[0])

    # Compute speeds for wrist and ankle joints
    for joint_name in wrist_joints + ankle_joints:
        if joint_name not in joints:
            print(f"Warning: Joint {joint_name} not found. Skipping.")
            continue

        # Extract joint angles
        joint_angles = extract_joint_angles_robust(joints, motion_data, joint_name)

        if joint_angles is None:
            continue

        # Compute derivative (speed) for each rotation channel
        for channel, angles in joint_angles.items():
            # Compute speed using numerical differentiation
            joint_speed = np.abs(np.gradient(angles) / frame_time)
            joint_speeds += joint_speed

    return joint_speeds


def segment_motion_trajectories(bvh_filename, motion_data, joints, frame_time,
                                target_joints=None,
                                wrist_joints=['LeftWrist', 'RightWrist'],
                                ankle_joints=['LeftAnkle', 'RightAnkle'],
                                min_boundary_distance=0.160):  # 160 ms
    """
    Segment motion trajectories based on joint speed and visualize full joint trajectories

    Args:
        bvh_filename: Name of BVH file for saving plot
        motion_data: Smoothed motion capture data
        joints: Joint mapping from BVH parser
        frame_time: Time between frames
        target_joints: List of joints to plot (if None, use wrist and ankle joints)
        wrist_joints: List of wrist joint names for speed computation
        ankle_joints: List of ankle joint names for speed computation
        min_boundary_distance: Minimum distance between boundaries in seconds

    Returns:
        Tuple of (segments, boundary_frames, joint_speeds)
    """
    # Set default target joints if not provided
    if target_joints is None:
        target_joints = wrist_joints + ankle_joints + ['Hip', 'Spine', 'Thorax']

    # Compute joint speeds
    joint_speeds = compute_joint_speed(motion_data, joints, frame_time,
                                       wrist_joints, ankle_joints)

    # Minimum distance in frames
    min_frames = int(min_boundary_distance / frame_time)

    # Find speed minima as potential segment boundaries
    peaks, _ = find_peaks(-joint_speeds, distance=min_frames, prominence=0.5)

    # Add start and end frames
    boundary_frames = [0] + list(peaks) + [len(joint_speeds) - 1]
    boundary_frames.sort()

    # Create segments
    segments = [boundary_frames[i:i + 2] for i in range(len(boundary_frames) - 1)]

    # Create time vector
    time_vector = np.arange(len(joint_speeds)) * frame_time

    # Create plots
    fig, axes = plt.subplots(len(target_joints), 1, figsize=(16, 5 * len(target_joints)))
    if len(target_joints) == 1:
        axes = [axes]

    # Color palette
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

    # Iterate through target joints
    for i, joint_name in enumerate(target_joints):
        # Skip if joint not in joints
        if joint_name not in joints:
            print(f"Warning: Joint {joint_name} not found. Skipping.")
            continue

        # Extract joint angles
        joint_angles = extract_joint_angles_robust(joints, motion_data, joint_name)

        if joint_angles is None:
            continue

        ax = axes[i]
        ax.set_title(f'{joint_name} Joint Angles with Motion Segments',
                     fontsize=16, fontweight='bold')

        # Plot each rotation channel
        for j, (channel, angle_data) in enumerate(joint_angles.items()):
            color = colors[j % len(colors)]
            ax.plot(time_vector, angle_data,
                    color=color,
                    label=f'{channel}',
                    linewidth=1.5,
                    alpha=0.7)

        # Plot segment boundaries
        for boundary in boundary_frames[1:-1]:  # Exclude first and last
            ax.axvline(x=time_vector[boundary], color='r', linestyle='--', alpha=0.7)

        # Highlight segments with different colors
        segment_colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        for j, segment in enumerate(segments):
            start_time = time_vector[segment[0]]
            end_time = time_vector[segment[1]]
            ax.axvspan(start_time, end_time, color=segment_colors[j], alpha=0.2,
                       label=f'Segment {j + 1}')

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time_vector[-1])

    plt.tight_layout()

    # Save plot
    figures_dir = os.path.join("./../../results", 'motion_segmentation')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"{bvh_filename}_joint_trajectories_segmentation.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Print segment information
    print("\nMotion Segments:")
    for i, segment in enumerate(segments, 1):
        print(f"Segment {i}: Frames {segment[0]}-{segment[1]} "
              f"(Time: {time_vector[segment[0]]:.2f}s - {time_vector[segment[1]]:.2f}s)")

    return segments, boundary_frames, joint_speeds


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


def calculate_power_spectrum(signal_data, sampling_rate, nperseg=None):
    """
    Calculate power spectral density using Welch's method

    Args:
        signal_data: 1D numpy array of time-domain signal
        sampling_rate: Sampling frequency in Hz
        nperseg: Length of each segment in welch's method (default: use signal length/8)

    Returns:
        frequencies: Frequency bins
        power_spectrum: Power spectral density
    """
    # If nperseg not specified, use default Welch recommendation
    if nperseg is None:
        nperseg = min(len(signal_data), max(256, len(signal_data) // 8))

    # Compute power spectral density
    frequencies, power_spectrum = welch(signal_data,
                                        fs=sampling_rate,
                                        nperseg=nperseg)

    return frequencies, power_spectrum


def analyze_filtering_impact(motion_data, joints, target_joints=None,
                             cutoff_frequencies=[3.0, 6.0, 10.0], filter_order=6):
    """
    Analyze power spectrum before and after filtering with different cutoff frequencies

    Args:
        motion_data: numpy array of motion data [frames, channels]
        joints: dict of joint mappings from BVH parser
        target_joints: list of joints to analyze (optional)
        cutoff_frequencies: list of cutoff frequencies to test
        filter_order: order of Butterworth filter

    Returns:
        Visualization of power spectra for each channel
    """
    # Set default target joints if not provided
    if target_joints is None:
        target_joints = ['LeftWrist', 'RightWrist', 'LeftKnee', 'RightKnee', 'LeftAnkle']

    # Calculate sampling frequency
    sampling_rate = 30
    nyquist_freq = sampling_rate / 2.0

    # Filter target joints to only those that exist
    valid_joints = [j for j in target_joints if j in joints]
    invalid_joints = [j for j in target_joints if j not in joints]

    if invalid_joints:
        print(f"\nSkipping non-existent joints: {invalid_joints}")

    if not valid_joints:
        print("No valid joints found to analyze")
        return None

    # Collect channel indices for valid joints
    joint_channels = {}
    for joint_name in valid_joints:
        joint_info = joints[joint_name]
        start_idx = joint_info['start_index']
        channels = joint_info['channels']

        # Store channel indices for this joint's rotations
        joint_channels[joint_name] = {
            'start_index': start_idx,
            'channel_indices': [start_idx + i for i in range(len(channels))
                                if 'rotation' in channels[i].lower()]
        }

    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(cutoff_frequencies)))

    # Process each channel
    for channel_idx in range(motion_data.shape[1]):
        # Create a new figure for this specific channel
        plt.figure(figsize=(10, 6))

        # Original signal power spectrum
        orig_signal = motion_data[:, channel_idx]
        orig_freq, orig_psd = calculate_power_spectrum(orig_signal, sampling_rate)

        # Plot original signal's power spectrum
        plt.semilogy(orig_freq, orig_psd,
                     label='Original Signal',
                     color='black')

        # Process for different cutoff frequencies
        for k, cutoff_freq in enumerate(cutoff_frequencies):
            # Validate and adjust cutoff frequency
            if cutoff_freq >= nyquist_freq:
                cutoff_freq = nyquist_freq * 0.8

            # Normalize cutoff frequency
            normalized_cutoff = cutoff_freq / nyquist_freq

            # Design Butterworth filter
            b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)

            # Apply filter
            filtered_signal = filtfilt(b, a, orig_signal)

            # Compute power spectrum of filtered signal
            filt_freq, filt_psd = calculate_power_spectrum(filtered_signal, sampling_rate)

            # Plot filtered signal's power spectrum
            plt.semilogy(filt_freq, filt_psd,
                         label=f'Filtered (Cutoff {cutoff_freq:.1f} Hz)',
                         color=colors[k])

            # Add vertical line at cutoff frequency
            plt.axvline(x=cutoff_freq, color='r', linestyle='--', alpha=0.5)

        # Try to get joint name for this channel
        channel_joint = "Unknown Channel"
        for joint_name, joint_info in joints.items():
            start_idx = joint_info['start_index']
            num_channels = len(joint_info['channels'])
            if start_idx <= channel_idx < start_idx + num_channels:
                channel_index_in_joint = channel_idx - start_idx
                channel_name = joint_info['channels'][channel_index_in_joint]
                channel_joint = f"{joint_name} {channel_name}"
                break

        # Formatting
        plt.title(f'Channel {channel_idx}: {channel_joint} Power Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency')
        plt.grid(True)
        plt.legend()

        # Save figure
        figures_dir = os.path.join("./../../results", 'power_spectrum')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir,
                                 f'channel_{channel_idx}_{channel_joint.replace(" ", "_")}_power_spectrum.png'),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()

    return None


def main():

    bvh_filename = "subject_71_motion_07"
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
    # print("\n Applying Butterworth smoothing and comparing with original data... ...")
    # result_smoothed = visualize_joint_angles_with_smoothing(
    #     bvh_filename,
    #     joints_to_analyze,
    #     apply_smoothing=True,
    #     cutoff_freq=3.0,  # 6 Hz as recommended by research
    #     filter_order=4
    # )

    joints, motion_data, frame_time, frames = parse_bvh_robust(f"../../data/bvh_files/{bvh_filename}.bvh")

    ### For signal cutoff frequenccy analysis and optimal choice

    analyze_filtering_impact(
        motion_data,
        joints,
        target_joints=joints_to_analyze,
        cutoff_frequencies=[3.0, 6.0, 10.0],
        filter_order=4
    )

    ### For segmentation visualization

    # # Optional: Apply smoothing
    # smoothed_motion_data = apply_butterworth_smoothing(motion_data)
    #
    # # Segment motion
    # segments, boundaries, speeds = segment_motion_trajectories(
    #     bvh_filename,
    #     smoothed_motion_data,
    #     joints,
    #     frame_time,
    #     target_joints=['LeftWrist', 'RightWrist', 'LeftKnee', 'RightKnee', 'Hip']  # Customize as needed
    # )

if __name__ == "__main__":
    main()