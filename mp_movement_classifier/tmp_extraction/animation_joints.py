import numpy as np
import re
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, welch
from scipy.signal import find_peaks
import warnings
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def parse_bvh_robust(file_path):
    """
    Robust BVH parser that handles various format issues
    """
    with open(file_path, 'r') as file:
        content = file.read()

    parts = content.split('MOTION')
    if len(parts) < 2:
        print("❌ Invalid BVH format: No MOTION section found")
        return None, None, None, None

    hierarchy = parts[0]
    motion_part = parts[1]

    joints = {}
    channel_index = 0

    joint_pattern = r'(ROOT|JOINT)\s+(\w+)'
    channel_pattern = r'CHANNELS\s+(\d+)\s+(.*)'

    lines = hierarchy.split('\n')
    current_joint = None

    for line in lines:
        line = line.strip()

        joint_match = re.search(joint_pattern, line)
        if joint_match:
            current_joint = joint_match.group(2)

        channel_match = re.search(channel_pattern, line)
        if channel_match and current_joint:
            num_channels = int(channel_match.group(1))
            channels = channel_match.group(2).split()

            joints[current_joint] = {
                'channels': channels,
                'start_index': channel_index
            }
            channel_index += num_channels

    motion_lines = motion_part.strip().split('\n')

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

    frame_data = []

    for line in motion_lines:
        line = line.strip()
        if line and not line.startswith('Frames') and not line.startswith('Frame Time'):
            try:
                values = [float(x) for x in line.split()]
                frame_data.extend(values)
            except ValueError:
                continue
    total_channels = sum(len(joint['channels']) for joint in joints.values())
    expected_data_points = total_channels * frames

    # if len(frame_data) < expected_data_points:
    #     print(f"⚠️ Warning: Less data than expected. Using available frames.")
    #     available_frames = len(frame_data) // total_channels
    #     motion_data = np.array(frame_data[:available_frames * total_channels]).reshape(available_frames, total_channels)
    #     frames = available_frames
    # else:
    motion_data = np.array(frame_data[:expected_data_points]).reshape(frames, total_channels)


    return joints, motion_data, frame_time, frames


def apply_butterworth_smoothing(motion_data, cutoff_freq=6.0, filter_order=6, sampling_freq=30):
    """
    Apply sixth-order Butterworth filter to motion capture data
    """
    nyquist_freq = sampling_freq / 2.0

    if cutoff_freq >= nyquist_freq:
        print(f"⚠️ Warning: Cutoff frequency ({cutoff_freq} Hz) is too high for sampling rate ({sampling_freq:.1f} Hz)")
        cutoff_freq = nyquist_freq * 0.8
        print(f"   Adjusting cutoff to {cutoff_freq:.1f} Hz")

    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    smoothed_data = np.zeros_like(motion_data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for channel in range(motion_data.shape[1]):
            smoothed_data[:, channel] = filtfilt(b, a, motion_data[:, channel])

    return smoothed_data


def compute_joint_speed(motion_data, joints, frame_time, wrist_joints=['LeftWrist', 'RightWrist'],
                        ankle_joints=['LeftAnkle', 'RightAnkle']):
    """
    Compute speed of specified joints
    """
    joint_speeds = np.zeros(motion_data.shape[0])

    for joint_name in wrist_joints + ankle_joints:
        if joint_name not in joints:
            print(f"Warning: Joint {joint_name} not found. Skipping.")
            continue

        joint_angles = extract_joint_angles_robust(joints, motion_data, joint_name)
        if joint_angles is None:
            continue

        for channel, angles in joint_angles.items():
            joint_speed = np.abs(np.gradient(angles) / frame_time)
            joint_speeds += joint_speed

    return joint_speeds


def segment_motion_trajectories(bvh_filename, motion_data, joints, frame_time,
                                target_joints=None,
                                wrist_joints=['LeftWrist', 'RightWrist'],
                                ankle_joints=['LeftAnkle', 'RightAnkle'],
                                min_boundary_distance=0.160):
    """
    Segment motion trajectories based on joint speed and visualize full joint trajectories
    """
    if target_joints is None:
        target_joints = wrist_joints + ankle_joints + ['Hip', 'Spine', 'Thorax']

    joint_speeds = compute_joint_speed(motion_data, joints, frame_time, wrist_joints, ankle_joints)
    min_frames = 30
    print(f"Minimum distance in frames: {min_frames}")
    peaks, _ = find_peaks(-joint_speeds, distance=min_frames)
    boundary_frames = [0] + list(peaks) + [len(joint_speeds) - 1]
    boundary_frames.sort()

    boundaries = [boundary_frames[i:i + 2] for i in range(len(boundary_frames) - 1)]
    segments = [motion_data[boundary_frames[i]:boundary_frames[i + 1], :] for i in range(len(boundary_frames) - 1)]
    time_vector = np.arange(len(joint_speeds)) * frame_time

    fig, axes = plt.subplots(len(target_joints), 1, figsize=(16, 5 * len(target_joints)))
    if len(target_joints) == 1:
        axes = [axes]

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

    for i, joint_name in enumerate(target_joints):
        if joint_name not in joints:
            print(f"Warning: Joint {joint_name} not found. Skipping.")
            continue

        joint_angles = extract_joint_angles_robust(joints, motion_data, joint_name)
        if joint_angles is None:
            continue

        ax = axes[i]
        ax.set_title(f'{joint_name} Joint Angles with Motion Segments', fontsize=16, fontweight='bold')

        for j, (channel, angle_data) in enumerate(joint_angles.items()):
            color = colors[j % len(colors)]
            ax.plot(time_vector, angle_data, color=color, label=f'{channel}', linewidth=1.5, alpha=0.7)

        for boundary in boundary_frames[1:-1]:
            ax.axvline(x=time_vector[boundary], color='r', linestyle='--', alpha=0.7)

        segment_colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        for j, segment in enumerate(segments):
            boundary = boundaries[j]
            start_time = time_vector[boundary[0]]
            end_time = time_vector[boundary[1]]
            ax.axvspan(start_time, end_time, color=segment_colors[j], alpha=0.2, label=f'Segment {j + 1}')

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time_vector[-1])

    plt.tight_layout()

    figures_dir = os.path.join("./../../results", 'motion_segmentation')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"{bvh_filename}_joint_trajectories_segmentation.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("\n📊 Motion Segments:")
    for i, segment in enumerate(segments, 1):
        boundary = boundaries[i - 1]
        print(f"   Segment {i}: Frames {boundary[0]}-{boundary[1]}")
        print(f"   Time: {time_vector[boundary[0]]:.2f}s - {time_vector[boundary[1]]:.2f}s")

    return segments, boundaries, boundary_frames, joint_speeds


def extract_joint_angles_robust(joints, motion_data, joint_name):
    """
    Extract rotation angles for a specific joint with error handling
    """
    if joint_name not in joints:
        return None

    joint_info = joints[joint_name]
    start_idx = joint_info['start_index']
    channels = joint_info['channels']

    angles = {}
    for i, channel in enumerate(channels):
        if 'rotation' in channel.lower():
            if start_idx + i < motion_data.shape[1]:
                angles[channel] = motion_data[:, start_idx + i]

    return angles if angles else None


def parse_bvh_hierarchy(file_path):
    """
    Parse BVH file to extract skeleton hierarchy and bone offsets
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    hierarchy = {}
    joint_stack = []
    current_joint = None

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith('ROOT') or line_stripped.startswith('JOINT'):
            parts = line_stripped.split()
            joint_name = parts[1]
            current_joint = joint_name

            parent = joint_stack[-1] if joint_stack else None
            hierarchy[joint_name] = {
                'parent': parent,
                'offset': [0.0, 0.0, 0.0],
                'channels': [],
                'children': []
            }

            # Add this joint as a child of its parent
            if parent:
                hierarchy[parent]['children'].append(joint_name)

            joint_stack.append(joint_name)

        elif line_stripped.startswith('OFFSET'):
            parts = line_stripped.split()
            if current_joint and len(parts) >= 4:
                offset = [float(parts[1]), float(parts[2]), float(parts[3])]
                hierarchy[current_joint]['offset'] = offset

        elif line_stripped.startswith('CHANNELS'):
            parts = line_stripped.split()
            num_channels = int(parts[1])
            channels = parts[2:2 + num_channels]
            if current_joint:
                hierarchy[current_joint]['channels'] = channels

        elif line_stripped == '}':
            if joint_stack:
                joint_stack.pop()
                current_joint = joint_stack[-1] if joint_stack else None

        if line_stripped == 'MOTION':
            break

    return hierarchy


def build_skeleton_connections(hierarchy):
    """
    Build skeleton connections from hierarchy (parent-child relationships)
    Returns list of (parent_name, child_name) tuples
    """
    connections = []

    for joint_name, joint_info in hierarchy.items():
        for child_name in joint_info['children']:
            connections.append((joint_name, child_name))

    return connections


def rotation_matrix_x(angle_deg):
    """Rotation matrix around X axis"""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rotation_matrix_y(angle_deg):
    """Rotation matrix around Y axis"""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotation_matrix_z(angle_deg):
    """Rotation matrix around Z axis"""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def get_rotation_matrix_from_channels(motion_data, joints, joint_name, frame_idx):
    """
    Get the rotation matrix for a joint at a specific frame
    """
    if joint_name not in joints:
        return np.eye(3)

    joint_info = joints[joint_name]
    start_idx = joint_info['start_index']
    channels = joint_info['channels']

    R = np.eye(3)

    # Apply rotations in the order specified by channels
    for i, channel in enumerate(channels):
        if 'rotation' in channel.lower():
            angle = motion_data[frame_idx, start_idx + i]
            axis = channel[0].upper()

            if axis == 'X':
                R = R @ rotation_matrix_x(angle)
            elif axis == 'Y':
                R = R @ rotation_matrix_y(angle)
            elif axis == 'Z':
                R = R @ rotation_matrix_z(angle)

    return R


def compute_joint_positions_recursive(motion_data, joints, hierarchy, frame_idx, joint_name,
                                      parent_position, parent_rotation):
    """
    Recursively compute joint positions using forward kinematics
    """
    if joint_name not in hierarchy:
        return {}

    positions = {}

    # Get joint info
    offset = np.array(hierarchy[joint_name]['offset'])

    # Get local rotation
    local_rotation = get_rotation_matrix_from_channels(motion_data, joints, joint_name, frame_idx)

    # Get local translation (only for root)
    local_translation = np.zeros(3)
    if joint_name in joints and 'Xposition' in joints[joint_name]['channels']:
        channels = joints[joint_name]['channels']
        start_idx = joints[joint_name]['start_index']
        x_idx = channels.index('Xposition')
        y_idx = channels.index('Yposition')
        z_idx = channels.index('Zposition')
        local_translation = np.array([
            motion_data[frame_idx, start_idx + x_idx],
            motion_data[frame_idx, start_idx + y_idx],
            motion_data[frame_idx, start_idx + z_idx]
        ])

    # Compute global transformation
    if parent_position is None:
        # Root joint
        global_position = local_translation
        global_rotation = local_rotation
    else:
        # Child joint: transform offset by parent's rotation, then add to parent's position
        global_position = parent_position + parent_rotation @ offset
        global_rotation = parent_rotation @ local_rotation

    positions[joint_name] = {
        'position': global_position,
        'rotation': global_rotation
    }

    # Recursively process children
    for child_name in hierarchy[joint_name]['children']:
        child_positions = compute_joint_positions_recursive(
            motion_data, joints, hierarchy, frame_idx,
            child_name, global_position, global_rotation
        )
        positions.update(child_positions)

    return positions


def compute_joint_positions(motion_data, joints, hierarchy, frame_idx):
    """
    Compute 3D positions of all joints for a specific frame using forward kinematics
    """
    # Find root joint (joint with no parent)
    root_joint = None
    for joint_name, joint_info in hierarchy.items():
        if joint_info['parent'] is None:
            root_joint = joint_name
            break

    if root_joint is None:
        print("❌ No root joint found!")
        return {}

    # Start recursive computation from root
    positions = compute_joint_positions_recursive(
        motion_data, joints, hierarchy, frame_idx,
        root_joint, None, np.eye(3)
    )

    return positions


def create_skeleton_animation(bvh_filename, motion_data, joints, hierarchy, frame_time,
                              fps=30, output_path=None):
    """
    Create and save a 3D animation of the skeleton motion
    """
    print("\n🎬 Creating skeleton animation...")

    # Build skeleton connections from hierarchy
    skeleton_connections = build_skeleton_connections(hierarchy)

    print(f"\n📊 Skeleton Structure:")
    print(f"   Joints: {list(hierarchy.keys())}")
    print(f"   Connections: {len(skeleton_connections)}")
    for parent, child in skeleton_connections:
        print(f"      {parent} -> {child}")

    num_frames = motion_data.shape[0]

    # Compute positions for all frames to get proper axis limits
    print("\n   Computing joint positions for axis limits...")
    all_positions = []
    sample_rate = max(1, num_frames // 50)

    for frame_idx in range(0, num_frames, sample_rate):
        positions = compute_joint_positions(motion_data, joints, hierarchy, frame_idx)
        for joint_name, data in positions.items():
            all_positions.append(data['position'])

    if len(all_positions) == 0:
        print("❌ No joint positions computed!")
        return None

    all_positions = np.array(all_positions)

    print(f"   Position range: X[{all_positions[:, 0].min():.1f}, {all_positions[:, 0].max():.1f}], "
          f"Y[{all_positions[:, 1].min():.1f}, {all_positions[:, 1].max():.1f}], "
          f"Z[{all_positions[:, 2].min():.1f}, {all_positions[:, 2].max():.1f}]")

    # Set up figure
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits with equal scaling
    x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
    y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
    z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
    max_range = max(x_range, y_range, z_range) * 1.2

    x_mid = (all_positions[:, 0].max() + all_positions[:, 0].min()) / 2
    y_mid = (all_positions[:, 1].max() + all_positions[:, 1].min()) / 2
    z_mid = (all_positions[:, 2].max() + all_positions[:, 2].min()) / 2

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'Skeleton Animation: {bvh_filename}', fontsize=14, fontweight='bold')

    # Initialize plot elements
    joint_plot, = ax.plot([], [], [], 'ro', markersize=6, label='Joints')
    lines = [ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.8)[0] for _ in skeleton_connections]
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        joint_plot.set_data([], [])
        joint_plot.set_3d_properties([])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        time_text.set_text('')
        return [joint_plot] + lines + [time_text]

    def update(frame_idx):
        positions = compute_joint_positions(motion_data, joints, hierarchy, frame_idx)

        if len(positions) == 0:
            return [joint_plot] + lines + [time_text]

        # Extract joint positions
        joint_names = list(positions.keys())
        joint_positions_array = np.array([positions[name]['position'] for name in joint_names])

        # Update joints
        joint_plot.set_data(joint_positions_array[:, 0], joint_positions_array[:, 1])
        joint_plot.set_3d_properties(joint_positions_array[:, 2])

        # Update skeleton lines (using joint names directly)
        for line, (parent_name, child_name) in zip(lines, skeleton_connections):
            if parent_name in positions and child_name in positions:
                parent_pos = positions[parent_name]['position']
                child_pos = positions[child_name]['position']

                x_data = [parent_pos[0], child_pos[0]]
                y_data = [parent_pos[1], child_pos[1]]
                z_data = [parent_pos[2], child_pos[2]]

                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)

        current_time = frame_idx * frame_time
        time_text.set_text(f'Time: {current_time:.2f}s | Frame: {frame_idx}/{num_frames}')

        return [joint_plot] + lines + [time_text]

    # Create animation
    frame_skip = max(1, num_frames // 200)
    frames_to_animate = range(0, num_frames, frame_skip)

    print(f"\n   Animation settings:")
    print(f"      Total frames: {num_frames}")
    print(f"      Frame skip: {frame_skip}")
    print(f"      Animation frames: {len(frames_to_animate)}")
    print(f"      FPS: {fps}")

    anim = FuncAnimation(fig, update, frames=frames_to_animate,
                         init_func=init, blit=False, interval=1000 * frame_time * frame_skip)

    # Save animation
    if output_path is None:
        output_dir = os.path.join("./../../results", 'animations')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{bvh_filename}_smoothed_animation.mp4")

    try:
        print(f"\n   Saving animation to: {output_path}")
        anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
        print(f"✅ Animation saved successfully!")
    except Exception as e:
        print(f"⚠️ Could not save as MP4: {e}")
        output_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(output_path, writer='pillow', fps=fps // 2)
            print(f"✅ Animation saved as GIF!")
        except Exception as e2:
            print(f"❌ Could not save animation: {e2}")
            output_path = None

    plt.close()
    return output_path


def main():
    print("=" * 80)
    print("MOTION CAPTURE PROCESSING AND ANIMATION")
    print("=" * 80)

    bvh_filename = "subject_12_motion_05"
    bvh_file = f"../../data/bvh_files/{bvh_filename}.bvh"

    print("\n📁 Parsing BVH file...")
    joints, motion_data, frame_time, frames = parse_bvh_robust(bvh_file)

    if joints is None:
        print("❌ Failed to parse BVH file")
        return

    print(f"✅ Parsed: {len(joints)} joints, {frames} frames")
    print(f"   Available joints: {list(joints.keys())}")

    print("\n🔧 Applying Butterworth smoothing...")
    smoothed_motion_data = apply_butterworth_smoothing(
        motion_data,
        cutoff_freq=6.0,
        filter_order=4,
        sampling_freq=1 / frame_time
    )
    print(f"✅ Smoothing complete")

    print("\n🦴 Parsing skeleton hierarchy...")
    hierarchy = parse_bvh_hierarchy(bvh_file)
    print(f"✅ Hierarchy parsed: {len(hierarchy)} joints")

    # Print hierarchy structure
    print("\n   Hierarchy structure:")
    for joint_name, joint_info in hierarchy.items():
        parent = joint_info['parent'] if joint_info['parent'] else 'None (ROOT)'
        print(f"      {joint_name}: parent={parent}, children={joint_info['children']}")

    print("\n🎥 Creating animation...")
    animation_path = create_skeleton_animation(
        bvh_filename=bvh_filename,
        motion_data=smoothed_motion_data,
        joints=joints,
        hierarchy=hierarchy,
        frame_time=frame_time,
        fps=30
    )

    if animation_path:
        print(f"\n✅ Animation created: {animation_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()