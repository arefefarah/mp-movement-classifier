import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Joint names from your skeleton
JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

# Define skeleton connections (bone structure)
SKELETON_CONNECTIONS = [
    # Spine chain
    ('Hip', 'Spine'),
    ('Spine', 'Thorax'),
    ('Thorax', 'Neck'),
    ('Neck', 'Head'),

    # Right leg
    ('Hip', 'RHip'),
    ('RHip', 'RKnee'),
    ('RKnee', 'RAnkle'),

    # Left leg
    ('Hip', 'LHip'),
    ('LHip', 'LKnee'),
    ('LKnee', 'LAnkle'),

    # Right arm
    ('Thorax', 'RShoulder'),
    ('RShoulder', 'RElbow'),
    ('RElbow', 'RWrist'),

    # Left arm
    ('Thorax', 'LShoulder'),
    ('LShoulder', 'LElbow'),
    ('LElbow', 'LWrist'),
]


def extract_joint_positions_from_bvh(joints, motion_data, frame_idx):
    """
    Extract 3D positions of joints from BVH data at a specific frame.

    This is a simplified version - for real BVH you'd need to:
    1. Apply hierarchical transformations
    2. Use offset values from hierarchy

    For now, we'll use a simplified approach focusing on angles.
    """
    positions = {}

    # For each joint, extract its position data
    for joint_name in JOINT_NAMES:
        if joint_name not in joints:
            continue

        joint_info = joints[joint_name]
        start_idx = joint_info['start_index']
        channels = joint_info['channels']

        # Extract position (Xposition, Yposition, Zposition) or rotation data
        pos = []
        for i, channel in enumerate(channels):
            if start_idx + i < motion_data.shape[1]:
                pos.append(motion_data[frame_idx, start_idx + i])

        positions[joint_name] = np.array(pos) if pos else np.zeros(3)

    return positions


def compute_forward_kinematics(joints, motion_data, frame_idx):
    """
    Compute 3D positions using forward kinematics.
    This is a simplified version that builds the skeleton hierarchy.
    """
    # Initialize positions dictionary
    positions_3d = {}

    # Default offsets (you may need to extract these from BVH OFFSET data)
    # These are approximate values in cm
    joint_offsets = {
        'Hip': np.array([0, 0, 0]),
        'Spine': np.array([0, 10, 0]),
        'Thorax': np.array([0, 20, 0]),
        'Neck': np.array([0, 15, 0]),
        'Head': np.array([0, 10, 0]),
        'RHip': np.array([10, 0, 0]),
        'RKnee': np.array([0, -40, 0]),
        'RAnkle': np.array([0, -40, 0]),
        'LHip': np.array([-10, 0, 0]),
        'LKnee': np.array([0, -40, 0]),
        'LAnkle': np.array([0, -40, 0]),
        'RShoulder': np.array([15, 0, 0]),
        'RElbow': np.array([0, -25, 0]),
        'RWrist': np.array([0, -25, 0]),
        'LShoulder': np.array([-15, 0, 0]),
        'LElbow': np.array([0, -25, 0]),
        'LWrist': np.array([0, -25, 0]),
    }

    # Start with Hip at origin
    current_pos = np.array([0.0, 0.0, 0.0])

    # Get Hip position if it has position channels
    if 'Hip' in joints:
        joint_info = joints['Hip']
        start_idx = joint_info['start_index']
        channels = joint_info['channels']

        for i, channel in enumerate(channels):
            if 'position' in channel.lower() and start_idx + i < motion_data.shape[1]:
                channel_name = channel.lower()
                value = motion_data[frame_idx, start_idx + i]
                if 'xposition' in channel_name:
                    current_pos[0] = value
                elif 'yposition' in channel_name:
                    current_pos[1] = value
                elif 'zposition' in channel_name:
                    current_pos[2] = value

    positions_3d['Hip'] = current_pos.copy()

    # Build skeleton hierarchy (simplified)
    # For each joint, add offset from parent
    hierarchy = {
        'Hip': ['Spine', 'RHip', 'LHip'],
        'Spine': ['Thorax'],
        'Thorax': ['Neck', 'RShoulder', 'LShoulder'],
        'Neck': ['Head'],
        'RHip': ['RKnee'],
        'RKnee': ['RAnkle'],
        'LHip': ['LKnee'],
        'LKnee': ['LAnkle'],
        'RShoulder': ['RElbow'],
        'RElbow': ['RWrist'],
        'LShoulder': ['LElbow'],
        'LElbow': ['LWrist'],
    }

    def build_chain(parent_name, parent_pos):
        if parent_name not in hierarchy:
            return

        for child_name in hierarchy[parent_name]:
            if child_name in joint_offsets:
                child_pos = parent_pos + joint_offsets[child_name]
                positions_3d[child_name] = child_pos
                build_chain(child_name, child_pos)

    build_chain('Hip', positions_3d['Hip'])

    return positions_3d


def create_segment_animation(bvh_filename, segment_idx, segment_frames,
                             joints, motion_data, frame_time,
                             output_dir="../../results/segment_animations"):
    """
    Create and save animation for a specific motion segment.

    Args:
        bvh_filename: Name of the BVH file
        segment_idx: Index of the segment
        segment_frames: [start_frame, end_frame] for this segment
        joints: Joint information from BVH parser
        motion_data: Motion data array
        frame_time: Time between frames
        output_dir: Directory to save animations
    """
    start_frame, end_frame = segment_frames
    num_frames = end_frame - start_frame + 1

    print(f"\n🎬 Creating animation for Segment {segment_idx}...")
    print(f"   Frames: {start_frame} to {end_frame} ({num_frames} frames)")
    print(f"   Duration: {(num_frames * frame_time):.2f} seconds")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Compute all positions for this segment
    all_positions = []
    for frame in range(start_frame, end_frame + 1):
        positions = compute_forward_kinematics(joints, motion_data, frame)
        all_positions.append(positions)

    # Get bounds for consistent scaling
    all_x = []
    all_y = []
    all_z = []
    for positions in all_positions:
        for pos in positions.values():
            all_x.append(pos[0])
            all_y.append(pos[1])
            all_z.append(pos[2])

    x_range = [min(all_x) - 20, max(all_x) + 20]
    y_range = [min(all_y) - 20, max(all_y) + 20]
    z_range = [min(all_z) - 20, max(all_z) + 20]

    # Initialize plot elements
    lines = []
    points = None

    def init():
        """Initialize animation"""
        ax.clear()
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_title(f'Motion Segment {segment_idx}\n'
                     f'File: {bvh_filename} | Frames: {start_frame}-{end_frame}',
                     fontsize=12, fontweight='bold')
        return []

    def update(frame_idx):
        """Update animation for each frame"""
        nonlocal lines, points

        # Clear previous frame
        ax.clear()
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')

        # Get positions for this frame
        positions = all_positions[frame_idx]

        # Calculate time
        current_time = frame_idx * frame_time
        total_time = num_frames * frame_time

        ax.set_title(f'Motion Segment {segment_idx}\n'
                     f'Frame: {start_frame + frame_idx}/{end_frame} | '
                     f'Time: {current_time:.2f}/{total_time:.2f}s',
                     fontsize=12, fontweight='bold')

        # Draw skeleton connections
        for joint1_name, joint2_name in SKELETON_CONNECTIONS:
            if joint1_name in positions and joint2_name in positions:
                pos1 = positions[joint1_name]
                pos2 = positions[joint2_name]

                ax.plot3D([pos1[0], pos2[0]],
                          [pos1[1], pos2[1]],
                          [pos1[2], pos2[2]],
                          'b-', linewidth=2, alpha=0.7)

        # Draw joint points
        joint_x = [pos[0] for pos in positions.values()]
        joint_y = [pos[1] for pos in positions.values()]
        joint_z = [pos[2] for pos in positions.values()]

        ax.scatter(joint_x, joint_y, joint_z,
                   c='red', marker='o', s=50, alpha=0.8)

        # Add grid
        ax.grid(True, alpha=0.3)

        return []

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_frames,
                         init_func=init, blit=False,
                         interval=frame_time * 1000,  # Convert to milliseconds
                         repeat=True)

    # Save animation
    output_path = os.path.join(output_dir,
                               f"{bvh_filename}_segment_{segment_idx:02d}.gif")

    print(f"   💾 Saving animation to: {output_path}")
    writer = PillowWriter(fps=int(1 / frame_time))
    anim.save(output_path, writer=writer)

    plt.close(fig)
    print(f"   ✅ Animation saved successfully!")

    return output_path


def create_all_segment_animations(bvh_filename, segments, joints, motion_data, frame_time):
    """
    Create animations for all segments and save summary information.

    Args:
        bvh_filename: Name of the BVH file
        segments: List of [start_frame, end_frame] pairs
        joints: Joint information from BVH parser
        motion_data: Motion data array
        frame_time: Time between frames
    """
    print("\n" + "=" * 80)
    print("🎥 CREATING SEGMENT ANIMATIONS")
    print("=" * 80)

    output_dir = os.path.join("../../results", "segment_animations", bvh_filename)
    os.makedirs(output_dir, exist_ok=True)

    # Create animations for each segment
    animation_paths = []
    for i, segment in enumerate(segments, 1):
        try:
            path = create_segment_animation(
                bvh_filename, i, segment,
                joints, motion_data, frame_time,
                output_dir
            )
            animation_paths.append(path)
        except Exception as e:
            print(f"   ❌ Error creating animation for segment {i}: {e}")
            continue

    # Create summary file
    summary_path = os.path.join(output_dir, "segment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Motion Segmentation Summary\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"File: {bvh_filename}.bvh\n")
        f.write(f"Total Segments: {len(segments)}\n")
        f.write(f"Frame Time: {frame_time:.4f} seconds\n\n")

        for i, segment in enumerate(segments, 1):
            start, end = segment
            duration = (end - start + 1) * frame_time
            f.write(f"Segment {i:2d}: Frames {start:4d}-{end:4d} | "
                    f"Duration: {duration:6.2f}s | "
                    f"Frames: {end - start + 1:4d}\n")

    print(f"\n📄 Summary saved to: {summary_path}")
    print(f"\n✅ All animations created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)

    return animation_paths


def visualize_segment_comparison(bvh_filename, segments, joint_speeds, frame_time,
                                 output_dir="../../results/segment_animations"):
    """
    Create a visualization showing all segments with speed profile.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    time_vector = np.arange(len(joint_speeds)) * frame_time

    # Plot speed profile
    ax.plot(time_vector, joint_speeds, 'b-', linewidth=1.5, label='Joint Speed')

    # Highlight segments with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))

    for i, segment in enumerate(segments):
        start_time = time_vector[segment[0]]
        end_time = time_vector[segment[1]]
        ax.axvspan(start_time, end_time, color=colors[i], alpha=0.3,
                   label=f'Segment {i + 1}')

        # Add segment number in the middle
        mid_time = (start_time + end_time) / 2
        ax.text(mid_time, ax.get_ylim()[1] * 0.95, f'S{i + 1}',
                ha='center', va='top', fontsize=12, fontweight='bold')

    # Mark boundaries
    for boundary in [seg[0] for seg in segments[1:]]:
        ax.axvline(x=time_vector[boundary], color='r', linestyle='--',
                   linewidth=2, alpha=0.7)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Joint Speed (deg/s)', fontsize=12)
    ax.set_title(f'Motion Segmentation Overview: {bvh_filename}\n'
                 f'{len(segments)} segments detected',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, bvh_filename,
                               f"{bvh_filename}_segmentation_overview.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Segmentation overview saved to: {output_path}")

    return output_path