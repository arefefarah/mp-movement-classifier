import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import os
from mpl_toolkits.mplot3d import Axes3D
import warnings
import re
import sys

from mp_movement_classifier.utils.utils import H36M_KEYPOINT_NAMES, SKELETON_CONNECTIONS


def parse_bvh_hierarchy(file_path):
    """
    Parse BVH file to extract hierarchy information including offsets and parent-child relationships
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Split into hierarchy and motion sections
    parts = content.split('MOTION')
    if len(parts) < 2:
        return None, None, None

    hierarchy_section = parts[0]
    motion_section = parts[1]

    # Parse hierarchy
    joints = {}
    joint_hierarchy = {}  # parent -> [children]
    joint_parents = {}  # child -> parent
    joint_offsets = {}  # joint -> offset vector
    joint_order = []  # order of joints as they appear

    lines = hierarchy_section.split('\n')
    current_joint = None
    joint_stack = []
    channel_index = 0

    for line in lines:
        line = line.strip()

        # Match ROOT or JOINT
        joint_match = re.match(r'(ROOT|JOINT)\s+(\w+)', line)
        if joint_match:
            current_joint = joint_match.group(2)
            joint_stack.append(current_joint)
            joint_order.append(current_joint)

            # Set parent-child relationships
            if len(joint_stack) > 1:
                parent = joint_stack[-2]
                joint_parents[current_joint] = parent
                if parent not in joint_hierarchy:
                    joint_hierarchy[parent] = []
                joint_hierarchy[parent].append(current_joint)
            else:
                joint_parents[current_joint] = None  # Root joint

        # Match OFFSET
        elif line.startswith('OFFSET') and current_joint:
            offset_values = line.split()[1:4]
            joint_offsets[current_joint] = np.array([float(x) for x in offset_values])

        # Match CHANNELS
        elif line.startswith('CHANNELS') and current_joint:
            channel_match = re.match(r'CHANNELS\s+(\d+)\s+(.*)', line)
            if channel_match:
                num_channels = int(channel_match.group(1))
                channels = channel_match.group(2).split()

                joints[current_joint] = {
                    'channels': channels,
                    'start_index': channel_index,
                    'num_channels': num_channels
                }
                channel_index += num_channels

        # Match closing brace
        elif line == '}' and joint_stack:
            joint_stack.pop()

    # Parse motion data
    motion_lines = motion_section.strip().split('\n')
    frames = 0
    frame_time = 0.0

    for line in motion_lines:
        if line.startswith('Frames:'):
            frames = int(line.split(':')[1].strip())
        elif line.startswith('Frame Time:'):
            frame_time = float(line.split(':')[1].strip())

    # Extract motion data
    motion_data = []
    for line in motion_lines:
        line = line.strip()
        if line and not line.startswith('Frames') and not line.startswith('Frame Time'):
            try:
                values = [float(x) for x in line.split()]
                motion_data.append(values)
            except ValueError:
                continue

    if motion_data:
        motion_data = np.array(motion_data)
    else:
        return None

    return {
        'joints': joints,
        'hierarchy': joint_hierarchy,
        'parents': joint_parents,
        'offsets': joint_offsets,
        'joint_order': joint_order,
        'motion_data': motion_data,
        'frame_time': frame_time,
        'frames': frames
    }


def compute_joint_positions(bvh_data, frame_idx):
    """
    Compute 3D joint positions using CORRECTED BVH forward kinematics
    """
    joints = bvh_data['joints']
    offsets = bvh_data['offsets']
    parents = bvh_data['parents']
    motion_data = bvh_data['motion_data']
    joint_order = bvh_data['joint_order']

    if frame_idx >= len(motion_data):
        return {}

    frame_data = motion_data[frame_idx]

    # Store global transformations and final positions
    joint_transforms = {}
    joint_positions = {}

    def create_rotation_matrix(rx, ry, rz, order='ZXY'):
        """Create rotation matrix from Euler angles in degrees"""
        # Convert to radians
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

        # Individual rotation matrices
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # BVH standard rotation order is ZXY
        if order == 'ZXY':
            return Rz @ Rx @ Ry
        else:
            return Rx @ Ry @ Rz

    def get_joint_local_transform(joint_name):
        """Get local transformation matrix for a joint"""
        # Initialize with identity
        local_transform = np.eye(4)

        # Get joint info
        if joint_name not in joints:
            return local_transform

        joint_info = joints[joint_name]
        channels = joint_info.get('channels', [])
        start_idx = joint_info.get('start_index', 0)

        # Extract channel values
        translations = [0.0, 0.0, 0.0]  # tx, ty, tz
        rotations = [0.0, 0.0, 0.0]  # rx, ry, rz

        for i, channel in enumerate(channels):
            if start_idx + i >= len(frame_data):
                continue

            value = frame_data[start_idx + i]
            channel_lower = channel.lower()

            if 'position' in channel_lower or 'translation' in channel_lower:
                if 'x' in channel_lower:
                    translations[0] = value
                elif 'y' in channel_lower:
                    translations[1] = value
                elif 'z' in channel_lower:
                    translations[2] = value

            elif 'rotation' in channel_lower:
                if 'x' in channel_lower:
                    rotations[0] = value
                elif 'y' in channel_lower:
                    rotations[1] = value
                elif 'z' in channel_lower:
                    rotations[2] = value

        # Apply translation (for root joint usually)
        local_transform[0:3, 3] = translations

        # Apply rotations if any exist
        if any(r != 0.0 for r in rotations):
            rotation_matrix = create_rotation_matrix(rotations[0], rotations[1], rotations[2])
            local_transform = local_transform @ rotation_matrix

        return local_transform

    def compute_global_transform(joint_name):
        """Compute global transformation matrix for joint"""
        if joint_name in joint_transforms:
            return joint_transforms[joint_name]

        # Get local transformation
        local_transform = get_joint_local_transform(joint_name)

        # Create offset transformation (bone length from parent to this joint)
        offset_transform = np.eye(4)
        if joint_name in offsets:
            offset_transform[0:3, 3] = offsets[joint_name]

        # Get parent transformation
        parent_name = parents.get(joint_name)
        if parent_name is None:
            # Root joint: apply local transform, then offset
            global_transform = local_transform @ offset_transform
        else:
            # Child joint: parent_transform * local_transform * offset
            parent_transform = compute_global_transform(parent_name)
            global_transform = parent_transform @ local_transform @ offset_transform

        joint_transforms[joint_name] = global_transform
        return global_transform

    # Compute positions for all joints
    for joint_name in joint_order:
        if joint_name in joints or joint_name in offsets:
            transform = compute_global_transform(joint_name)
            # Extract position from transformation matrix
            joint_positions[joint_name] = transform[0:3, 3].copy()

    return joint_positions


def debug_bvh_structure(bvh_data):
    """Debug function to analyze BVH structure"""
    print("🔍 BVH Structure Analysis:")
    print(f"   Joint order: {bvh_data['joint_order']}")
    print(f"   Parent relationships:")
    for joint, parent in bvh_data['parents'].items():
        print(f"     {joint} -> parent: {parent}")

    print(f"   Offsets:")
    for joint, offset in bvh_data['offsets'].items():
        print(f"     {joint}: {offset}")

    print(f"   Channels:")
    for joint, info in bvh_data['joints'].items():
        print(f"     {joint}: {info['channels']} (index: {info['start_index']})")


def create_bvh_animation(bvh_file_path, start_frame=0, end_frame=None,
                         output_path=None, format='mp4', title_prefix=""):
    """
    Create animation from BVH file with CORRECTED forward kinematics
    """
    # Parse BVH file
    bvh_data = parse_bvh_hierarchy(bvh_file_path)
    if bvh_data is None:
        print("❌ Failed to parse BVH file")
        return None

    motion_data = bvh_data['motion_data']
    frame_time = bvh_data['frame_time']

    print(f"📋 BVH Skeleton Info:")
    print(f"   Joints: {bvh_data['joint_order']}")
    print(f"   Total joints: {len(bvh_data['joint_order'])}")

    # DEBUG: Show BVH structure
    debug_bvh_structure(bvh_data)

    # Set frame range
    if end_frame is None:
        end_frame = len(motion_data) - 1

    start_frame = max(0, start_frame)
    end_frame = min(len(motion_data) - 1, end_frame)
    num_frames = end_frame - start_frame + 1

    print(f"🎬 Creating animation: frames {start_frame}-{end_frame} ({num_frames} frames)")

    # DEBUG: Check first few frames
    print("\n🔍 DEBUG: Checking first frame joint positions...")
    first_frame_positions = compute_joint_positions(bvh_data, start_frame)
    if first_frame_positions:
        print("   First frame positions:")
        for joint, pos in first_frame_positions.items():
            print(f"     {joint}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        # Check anatomical consistency
        print("\n   🔍 Anatomical checks:")
        if 'Hip' in first_frame_positions and 'RightHip' in first_frame_positions:
            hip_pos = first_frame_positions['Hip']
            rhip_pos = first_frame_positions['RightHip']
            distance = np.linalg.norm(rhip_pos - hip_pos)
            print(f"     Hip to RightHip distance: {distance:.3f}")

        if 'RightHip' in first_frame_positions and 'RightKnee' in first_frame_positions:
            rhip_pos = first_frame_positions['RightHip']
            rknee_pos = first_frame_positions['RightKnee']
            distance = np.linalg.norm(rknee_pos - rhip_pos)
            print(f"     RightHip to RightKnee distance: {distance:.3f}")

    else:
        print("   ❌ No positions computed for first frame!")
        return None

    # Compute all positions (limit for testing)
    test_frames = min(50, num_frames)
    all_positions = []
    for i in range(test_frames):
        frame = start_frame + i
        if frame % 10 == 0:
            print(f"   Computing frame {frame}")

        positions = compute_joint_positions(bvh_data, frame)
        if positions:
            all_positions.append(positions)

    if not all_positions:
        print("❌ No valid joint positions computed")
        return None

    print(f"📐 Successfully computed positions for {len(all_positions)} frames")

    # Compute bounds
    all_x, all_y, all_z = [], [], []
    for positions in all_positions:
        for joint_name, pos in positions.items():
            if not (np.isnan(pos).any() or np.isinf(pos).any()):
                all_x.append(pos[0])
                all_y.append(pos[1])
                all_z.append(pos[2])

    if not all_x:
        print("❌ No valid position data found")
        return None

    # Calculate proper bounds
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)

    # Add reasonable margins
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    margin_x = max(x_range * 0.2, 0.3)
    margin_y = max(y_range * 0.2, 0.3)
    margin_z = max(z_range * 0.2, 0.3)

    x_bounds = [x_min - margin_x, x_max + margin_x]
    y_bounds = [y_min - margin_y, y_max + margin_y]
    z_bounds = [z_min - margin_z, z_max + margin_z]

    print(f"🔍 Position ranges:")
    print(f"   X: {x_min:.2f} to {x_max:.2f} → bounds: [{x_bounds[0]:.2f}, {x_bounds[1]:.2f}]")
    print(f"   Y: {y_min:.2f} to {y_max:.2f} → bounds: [{y_bounds[0]:.2f}, {y_bounds[1]:.2f}]")
    print(f"   Z: {z_min:.2f} to {z_max:.2f} → bounds: [{z_bounds[0]:.2f}, {z_bounds[1]:.2f}]")

    # Create animation
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_zlim(z_bounds)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        positions = all_positions[frame_idx]
        current_time = (start_frame + frame_idx) * frame_time

        ax.set_title(f'{title_prefix}BVH Motion (CORRECTED)\n'
                     f'Frame: {start_frame + frame_idx + 1} | Time: {current_time:.2f}s',
                     fontsize=12, fontweight='bold')

        # Draw skeleton based on actual BVH hierarchy
        connections_drawn = 0
        for joint_name in bvh_data['joint_order']:
            parent_name = bvh_data['parents'].get(joint_name)
            if parent_name and joint_name in positions and parent_name in positions:
                pos1 = positions[parent_name]  # Parent position
                pos2 = positions[joint_name]  # Child position

                # Draw bone
                ax.plot3D([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                          'b-', linewidth=2.5, alpha=0.8)
                connections_drawn += 1

        # Draw joints
        if positions:
            for i, (joint_name, pos) in enumerate(positions.items()):
                color = 'red' if 'Hip' in joint_name else 'blue'
                size = 100 if joint_name == 'Hip' else 60
                ax.scatter([pos[0]], [pos[1]], [pos[2]],
                           c=color, s=size, alpha=0.9,
                           edgecolors='black', linewidth=1)

                # Label key joints
                if joint_name in ['Hip', 'Neck', 'LeftWrist', 'RightWrist']:
                    ax.text(pos[0], pos[1], pos[2], joint_name, fontsize=8)

        if frame_idx == 0:
            print(f"🔍 Frame 0: Drew {connections_drawn} skeleton connections")

        ax.grid(True, alpha=0.3)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=10, azim=45)

        return []

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(all_positions),
                         interval=200, repeat=True, blit=False)

    # Save or show
    if output_path:
        try:
            writer = FFMpegWriter(fps=15, bitrate=2000) if format.lower() == 'mp4' else PillowWriter(fps=15)
            print(f"💾 Saving animation to: {output_path}")
            anim.save(output_path, writer=writer)
            print("✅ Animation saved successfully!")
            plt.close(fig)
            return output_path
        except Exception as e:
            print(f"❌ Error saving animation: {e}")
            plt.close(fig)
            return None
    else:
        plt.show()
        return anim


def create_all_segment_animations(bvh_filename, boundaries, joints, motion_data, frame_time):
    """
    Create animations using proper BVH parsing for complete sequence and segments
    """
    print("\n" + "=" * 80)
    print("🎥 CREATING SEGMENT ANIMATIONS WITH PROPER BVH PARSING")
    print("=" * 80)

    # Reconstruct BVH file path
    bvh_file_path = f"../../data/bvh_files/{bvh_filename}.bvh"
    if not os.path.exists(bvh_file_path):
        print(f"❌ BVH file not found: {bvh_file_path}")
        return []

    output_dir = os.path.join("../../results", "segment_animations", bvh_filename)
    os.makedirs(output_dir, exist_ok=True)

    animation_paths = []

    # FIRST: Create complete sequence animation
    print("\n🎬 Creating COMPLETE SEQUENCE animation...")
    complete_output_path = os.path.join(output_dir, f"{bvh_filename}_complete_sequence.mp4")

    complete_path = create_bvh_animation(
        bvh_file_path=bvh_file_path,
        start_frame=0,
        end_frame=None,
        output_path=complete_output_path,
        format='mp4',
        title_prefix="Complete Sequence - "
    )

    if complete_path:
        animation_paths.append(complete_path)

    # SECOND: Create individual segment animations
    print(f"\n🎬 Creating individual segment animations...")
    for i, boundary in enumerate(boundaries, 1):
        try:
            start_frame, end_frame = boundary[0], boundary[1]
            segment_output_path = os.path.join(output_dir, f"{bvh_filename}_segment_{i:02d}.mp4")

            segment_path = create_bvh_animation(
                bvh_file_path=bvh_file_path,
                start_frame=start_frame,
                end_frame=end_frame,
                output_path=segment_output_path,
                format='mp4',
                title_prefix=f"Segment {i} - "
            )

            if segment_path:
                animation_paths.append(segment_path)

        except Exception as e:
            print(f"   ❌ Error creating animation for segment {i}: {e}")
            continue

    # Create summary file
    summary_path = os.path.join(output_dir, "segment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Motion Segmentation Summary\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"File: {bvh_filename}.bvh\n")
        f.write(f"Total Segments: {len(boundaries)}\n")
        f.write(f"Frame Time: {frame_time:.4f} seconds\n\n")
        f.write(f"Complete Sequence: {complete_output_path}\n\n")

        for i, boundary in enumerate(boundaries, 1):
            start, end = boundary[0], boundary[1]
            duration = (end - start) * frame_time
            f.write(f"Segment {i:2d}: Frames {start:4d}-{end:4d} | "
                    f"Duration: {duration:6.2f}s | "
                    f"Frames: {end - start + 1:4d}\n")

    print(f"\n📄 Summary saved to: {summary_path}")
    print(f"\n✅ All animations created successfully!")
    print(f"📁 Complete sequence: {complete_output_path}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)

    return animation_paths


def visualize_segment_comparison(bvh_filename, boundaries, joint_speeds, frame_time,
                                 output_dir="../../results/segment_animations"):
    """
    Create a visualization showing all segments with speed profile.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    time_vector = np.arange(len(joint_speeds)) * frame_time

    # Plot speed profile
    ax.plot(time_vector, joint_speeds, 'b-', linewidth=1.5, label='Joint Speed')

    # Highlight boundaries with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(boundaries)))

    for i, boundary in enumerate(boundaries):
        start_time = time_vector[boundary[0]]
        end_time = time_vector[boundary[1]]
        ax.axvspan(start_time, end_time, color=colors[i], alpha=0.3,
                   label=f'Segment {i + 1}')

        # Add segment number in the middle
        mid_time = (start_time + end_time) / 2
        ax.text(mid_time, ax.get_ylim()[1] * 0.95, f'S{i + 1}',
                ha='center', va='top', fontsize=12, fontweight='bold')

    # Mark boundaries
    for boundary in [seg[0] for seg in boundaries[1:]]:
        ax.axvline(x=time_vector[boundary], color='r', linestyle='--',
                   linewidth=2, alpha=0.7)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Joint Speed (deg/s)', fontsize=12)
    ax.set_title(f'Motion Segmentation Overview: {bvh_filename}\n'
                 f'{len(boundaries)} segments detected',
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