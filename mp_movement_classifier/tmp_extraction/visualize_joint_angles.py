import numpy as np
import re
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, welch
from scipy.signal import find_peaks
import warnings
from matplotlib.animation import FuncAnimation

# Import animation functions
# from motion_segment_animator import (
#     create_all_segment_animations,
#     visualize_segment_comparison,
# )

from mp_movement_classifier.utils.utils import H36M_KEYPOINT_NAMES, SKELETON_CONNECTIONS
from mp_movement_classifier.utils import config


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


def apply_butterworth_smoothing(motion_data, cutoff_freq=6.0, filter_order=6, sampling_freq=30):
    """
    Apply sixth-order Butterworth filter to motion capture data

    Returns:
        smoothed_data: filtered motion data
    """
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

    Returns:
        Tuple of (segments, boundary_frames, joint_speeds)
    """
    # Set default target joints if not provided
    if target_joints is None:
        target_joints = wrist_joints + ankle_joints + ['Hip', 'Spine', 'Thorax']

    # Compute joint speeds
    joint_speeds = compute_joint_speed(motion_data, joints, frame_time,
                                       wrist_joints, ankle_joints)
    joint_speeds = compute_joint_speed(motion_data, joints, frame_time,
                                       wrist_joints, ankle_joints)
    min_frames = int(min_boundary_distance / frame_time)
    # min_frames = 30  # i manually change it to 6 instead of 4
    print(f"Minimum distance in frames: {min_frames}")
    peaks, _ = find_peaks(-joint_speeds, distance=min_frames)
    boundary_frames = [0] + list(peaks) + [len(joint_speeds) - 1]
    # print(f"boundary_frames: {boundary_frames}")
    boundary_frames.sort()

    # Create segments
    boundaries = [boundary_frames[i:i + 2] for i in range(len(boundary_frames) - 1)]
    segments = [motion_data[boundary_frames[i]:boundary_frames[i + 1], :] for i in range(len(boundary_frames) - 1)]

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
            boundary = boundaries[j]
            start_time = time_vector[boundary[0]]
            end_time = time_vector[boundary[1]]
            ax.axvspan(start_time, end_time, color=segment_colors[j], alpha=0.2,
                       label=f'Segment {j + 1}')

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time_vector[-1])

    plt.tight_layout()

    # Save plot
    figures_dir = os.path.join("./../../results", 'motion_segmentation')
    model_dir = os.path.join(config.SAVING_DIR, f"expmap_mp_model_20")
    figures_dir = os.path.join(model_dir, "motion_segmentation")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"{bvh_filename}_joint_trajectories_segmentation.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Print segment information
    # print(f"Duration of complete video : {len(joint_speeds) * frame_time} seconds")
    # print(f"Number of segments: {len(segments)}")
    print("\n📊 Motion Segments:")
    for i, segment in enumerate(segments, 1):
        boundary = boundaries[i - 1]
        print("segment shape", segment.shape)
        print("boundary", boundary)
        print(f"   Segment {i}: Frames {boundary[0]}-{boundary[1]} ")
        print(f"   Time: {time_vector[boundary[0]]} s - {time_vector[boundary[1]]} s")

    return segments, boundaries,boundary_frames, joint_speeds


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


def set_axes_equal(ax):
    """Set 3D axes to equal scale for better visualization."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = max(x_limits) - min(x_limits)
    y_range = max(y_limits) - min(y_limits)
    z_range = max(z_limits) - min(z_limits)
    max_range = max(x_range, y_range, z_range)
    mid_x = sum(x_limits) / 2
    mid_y = sum(y_limits) / 2
    mid_z = sum(z_limits) / 2
    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])



def segment_expmap_csv(file_name,csv_file_path, wrist_joints , ankle_joints):

    motion_df = pd.read_csv(csv_file_path)
    joint_speeds=0
    for joint_name in wrist_joints + ankle_joints:
        columns = [col for col in motion_df.columns if col.startswith(joint_name)]

        selected_df = motion_df[columns]
        rot_vec = selected_df.to_numpy() # 3 values of joint_name
        joint_speed = calculate_joint_angular_speed(rot_vec)
        joint_speeds += joint_speed

    min_boundary_distance = 1 #1 second for now
    frame_rate = 30
    frame_time = 1 / frame_rate
    min_frames = int(min_boundary_distance *frame_rate)
    # min_frames = 30  # i manually change it to 6 instead of 4
    print(f"Minimum distance in frames: {min_frames}")
    peaks, _ = find_peaks(-joint_speeds, distance=min_frames)
    boundary_frames = [0] + list(peaks) + [len(joint_speeds) - 1]
    # print(f"boundary_frames: {boundary_frames}")
    boundary_frames.sort()

    boundaries = [boundary_frames[i:i + 2] for i in range(len(boundary_frames) - 1)]
    segments = [motion_df.iloc[boundary[0]:boundary[1], :] for boundary in boundaries]
    print(f"len of segments: {len(segments)}")

    # # Create time vector
    time_vector = np.arange(motion_df.shape[0]) * frame_time

    target_joints=["LWrist","LKnee","LElbow","LAnkle","Head","LShoulder"]

    # Create plots
    fig, axes = plt.subplots(len(target_joints), 1, figsize=(16, 5 * len(target_joints)))
    if len(target_joints) == 1:
        axes = [axes]

    # Color palette
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

    # Iterate through target joints
    for i, joint_name in enumerate(target_joints):
        columns = [col for col in motion_df.columns if col.startswith(joint_name)]
        axis_angle_rep = motion_df[columns]
        print(f"axis_angle_rep shape: {axis_angle_rep.shape}")

        ax = axes[i]
        ax.set_title(f'{joint_name} Joint rep with Motion Segments',
                     fontsize=16, fontweight='bold')

        # Plot each rotation channel
        for idx,column in enumerate(columns):
            color = colors[idx % len(colors)]
            ax.plot(time_vector,motion_df[column],
                    color=color,
                    label=f'{column}',
                    linewidth=1.5,
                    alpha=0.7)

        # Plot segment boundaries
        for boundary in boundary_frames[1:-1]:  # Exclude first and last
            ax.axvline(x=time_vector[boundary], color='r', linestyle='--', alpha=0.7)

        # Highlight segments with different colors
        segment_colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        for j, segment in enumerate(segments):
            boundary = boundaries[j]
            start_time = time_vector[boundary[0]]
            end_time = time_vector[boundary[1]]
            ax.axvspan(start_time, end_time, color=segment_colors[j], alpha=0.2,
                       label=f'Segment {j + 1}')

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time_vector[-1])

    plt.tight_layout()

    figures_dir = os.path.join("./../../results", 'exp_map_segments')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"{file_name}_joint_trajectories_segmentation.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    return segments,boundaries


def calculate_joint_angular_speed(rotation_vectors, frame_rate=30):
    """
    Calculate angular speed from rotation vectors (exponential maps)

    Args:
        rotation_vectors: [num_frames, 3] array of rotation vectors
                         (one 3D rotation vector per frame)
        frame_rate: frames per second (Hz)

    Returns:
        angular_speeds: [num_frames-1] array of angular speeds (radians/second)
    """

    dt = 1.0 / frame_rate  # Time between frames

    # Get magnitude (angle) of each rotation vector
    rotation_angles = np.linalg.norm(rotation_vectors, axis=1)  # [num_frames]

    # Compute angular speed between consecutive frames
    angular_speeds = np.abs(np.diff(rotation_angles)) / dt

    return angular_speeds


def main():

    # Configuration
    filename = "subject_12_motion_05"
    bvh_file = f"../../data/expmap_bvh_files/{filename}.bvh"
    csv_file_path = f"../../data/expmap_csv_files/{filename}.csv"

    # joints, motion_data, frame_time, frames = parse_bvh_robust(bvh_file)
    # print("updated nume of joints: ", len(joints))

    segments,boundaries = segment_expmap_csv(filename,csv_file_path,wrist_joints=['LWrist', 'RWrist'],
                                             ankle_joints=['LAnkle', 'RAnkle'])


    # segments, boundaries, boundary_frames,speeds = segment_motion_trajectories(
    #     bvh_filename,
    #     motion_data,
    #     joints,
    #     frame_time,
    #     target_joints=['LeftAnkle', 'RightWrist', 'LeftKnee', 'RightKnee', 'RightHip'],
    #     min_boundary_distance=1
    # )
    # print(f"   ✅ Found {len(segments)} motion segments")
    #
    # animation_paths = create_all_segment_animations(
    #     bvh_filename,
    #     boundaries,
    #     joints,
    #     smoothed_motion_data,
    #     frame_time
    # )
    #
    # overview_path = visualize_segment_comparison(
    #     bvh_filename,
    #     boundaries,
    #     speeds,
    #     frame_time
    # )

if __name__ == "__main__":
    main()
