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
import plotly.graph_objects as go
import plotly.io as pio
import io
from PIL import Image
import json
import imageio.v2 as imageio
from matplotlib.animation import FuncAnimation

from mp_movement_classifier.utils.utils import calculate_angular_velocity_quat,segment_motion_csv
from mp_movement_classifier.utils import config
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import from_root_positions, fk
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import webbrowser
import os
from pymotion.render.viewer import Viewer

from mp_movement_classifier.utils.h36m_csv_converter import H36MConverter


def run_patched(self, debug=True, use_reloader=None):
    """Patched run method for Dash 3.x compatibility with layout check"""
    if use_reloader is None:
        use_reloader = self.use_reloader

    figure = self._create_figure()
    frames = [self._update_figure(frame) for frame in range(self.max_frames)]
    self._update_layout(figure, frames)
    self._set_up_callbacks()

    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://localhost:8050")

    # Use app.run() instead of app.run_server()
    self.app.run(debug=debug, use_reloader=use_reloader)

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
#
# def segment_motion_trajectories(bvh_filename, motion_data, joints, frame_time,
#                                 target_joints=None,
#                                 wrist_joints=['LeftWrist', 'RightWrist'],
#                                 ankle_joints=['LeftAnkle', 'RightAnkle'],
#                                 min_boundary_distance=0.160):  # 160 ms
#     """
#     Segment motion trajectories based on joint speed and visualize full joint trajectories
#
#     Returns:
#         Tuple of (segments, boundary_frames, joint_speeds)
#     """
#     # Set default target joints if not provided
#     if target_joints is None:
#         target_joints = wrist_joints + ankle_joints + ['Hip', 'Spine', 'Thorax']
#
#     # Compute joint speeds
#     joint_speeds = compute_joint_speed(motion_data, joints, frame_time,
#                                        wrist_joints, ankle_joints)
#     joint_speeds = compute_joint_speed(motion_data, joints, frame_time,
#                                        wrist_joints, ankle_joints)
#     min_frames = int(min_boundary_distance / frame_time)
#     # min_frames = 30  # i manually change it to 6 instead of 4
#     print(f"Minimum distance in frames: {min_frames}")
#     peaks, _ = find_peaks(-joint_speeds, distance=min_frames)
#     boundary_frames = [0] + list(peaks) + [len(joint_speeds) - 1]
#     # print(f"boundary_frames: {boundary_frames}")
#     boundary_frames.sort()
#
#     # Create segments
#     boundaries = [boundary_frames[i:i + 2] for i in range(len(boundary_frames) - 1)]
#     segments = [motion_data[boundary_frames[i]:boundary_frames[i + 1], :] for i in range(len(boundary_frames) - 1)]
#
#     # Create time vector
#     time_vector = np.arange(len(joint_speeds)) * frame_time
#
#     # Create plots
#     fig, axes = plt.subplots(len(target_joints), 1, figsize=(16, 5 * len(target_joints)))
#     if len(target_joints) == 1:
#         axes = [axes]
#
#     # Color palette
#     colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
#
#     # Iterate through target joints
#     for i, joint_name in enumerate(target_joints):
#         # Skip if joint not in joints
#         if joint_name not in joints:
#             print(f"Warning: Joint {joint_name} not found. Skipping.")
#             continue
#
#         # Extract joint angles
#         joint_angles = extract_joint_angles_robust(joints, motion_data, joint_name)
#
#         if joint_angles is None:
#             continue
#
#         ax = axes[i]
#         ax.set_title(f'{joint_name} Joint Angles with Motion Segments',
#                      fontsize=16, fontweight='bold')
#
#         # Plot each rotation channel
#         for j, (channel, angle_data) in enumerate(joint_angles.items()):
#             color = colors[j % len(colors)]
#             ax.plot(time_vector, angle_data,
#                     color=color,
#                     label=f'{channel}',
#                     linewidth=1.5,
#                     alpha=0.7)
#
#         # Plot segment boundaries
#         for boundary in boundary_frames[1:-1]:  # Exclude first and last
#             ax.axvline(x=time_vector[boundary], color='r', linestyle='--', alpha=0.7)
#
#         # Highlight segments with different colors
#         segment_colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
#         for j, segment in enumerate(segments):
#             boundary = boundaries[j]
#             start_time = time_vector[boundary[0]]
#             end_time = time_vector[boundary[1]]
#             ax.axvspan(start_time, end_time, color=segment_colors[j], alpha=0.2,
#                        label=f'Segment {j + 1}')
#
#         ax.set_xlabel('Time (seconds)', fontsize=12)
#         ax.set_ylabel('Angle (degrees)', fontsize=12)
#         ax.legend(fontsize=10, loc='upper right')
#         ax.grid(True, alpha=0.3)
#         ax.set_xlim(0, time_vector[-1])
#
#     plt.tight_layout()
#
#     # Save plot
#     # figures_dir = os.path.join("./../../result/tmp_configs/expmap_mp_model_20", 'motion_segmentation')
#     model_dir = os.path.join(config.SAVING_DIR, f"expmap_mp_model_20")
#     figures_dir = os.path.join(model_dir, "motion_segmentation")
#     os.makedirs(figures_dir, exist_ok=True)
#     plt.savefig(os.path.join(figures_dir, f"{bvh_filename}_joint_trajectories_segmentation.png"),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # Print segment information
#     # print(f"Duration of complete video : {len(joint_speeds) * frame_time} seconds")
#     # print(f"Number of segments: {len(segments)}")
#     print("\n📊 Motion Segments:")
#     for i, segment in enumerate(segments, 1):
#         boundary = boundaries[i - 1]
#         print("segment shape", segment.shape)
#         print("boundary", boundary)
#         print(f"   Segment {i}: Frames {boundary[0]}-{boundary[1]} ")
#         print(f"   Time: {time_vector[boundary[0]]} s - {time_vector[boundary[1]]} s")
#
#     return segments, boundaries,boundary_frames, joint_speeds


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


def visualize_motion_with_segmentation(file_name,csv_file_path, wrist_joints , ankle_joints,save_dir):

    motion_df = pd.read_csv(csv_file_path)
    frame_rate = 30
    frame_time = 1 / frame_rate
    segments, boundaries = segment_motion_csv(csv_file_path, data_type= "position",
                                              wrist_joints = wrist_joints,
                                              ankle_joints = ankle_joints,
                                              filtering=False)
    print(f"len of segments: {len(segments)}")
    boundary_frames = [boundaries[0][0]] + [b[1] for b in boundaries]


    # # Create time vector
    time_vector = np.arange(motion_df.shape[0]) * frame_time

    target_joints=["LWrist","LKnee","LElbow","LAnkle","Neck","LShoulder"]

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
            # ax.axvline(x=time_vector[boundary], color='r', linestyle='--', alpha=0.7)
            ax.axvline(x=time_vector[int(boundary)], color='r', linestyle='--', alpha=0.7)

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

    figures_dir = os.path.join(save_dir, "motion_segmentation")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"{file_name}.png"),
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
    joint_names = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Thorax', 'Neck',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
    ]
    model_dir = os.path.join("./../../results/tmp_configs", f"pymotion_position_mp_model_20")
    figures_dir = os.path.join("./../../results/segmentation_analysis")
    Path(figures_dir).mkdir(exist_ok=True)
    # for i in range(9):
    filename = f"subject_1_motion_02"
    csv_file = f"../../data/pymotion_quat_csv_files/{filename}.csv"
    bvh_reference = f"../../data/bvh_files/{filename}.bvh"
    MAPPING_FILE = "../../data/common_motion_mapping.json"
    motion_id_str = filename.split('_')[-1]
    with open(MAPPING_FILE, 'r') as f:
            data = json.load(f)
            motion_mapping = data["mapping"]
    id_to_motion_name = {id_val: motion_name for motion_name, id_val in motion_mapping.items()}
    motion_name = id_to_motion_name.get(int(motion_id_str))
    print(motion_name)

    bvh = BVH()
    bvh.load(bvh_reference)  # load euler angle rep from bvh data
    local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
    global_positions = local_positions[:, 0, :]  # root joint
    pos, rotmats = fk(local_rotations, global_positions, offsets, parents)

    # converter = H36MConverter() use converter to save csv file in future
    # Create the DataFrame for positions
    columns = []
    data = []
    for joint_idx, joint_name in enumerate(joint_names):
        columns.append(joint_name + "_x")
        columns.append(joint_name + "_y")
        columns.append(joint_name + "_z")

    for frame, pose in enumerate(pos):
        data_frame = []
        for joint_idx, joint_name in enumerate(joint_names):
            data_frame.extend(pose[joint_idx, :])
        data.append(data_frame)

    df = pd.DataFrame(data, columns=columns)
    csv_files_dir = Path(figures_dir) / "position_csv_files"
    os.makedirs(csv_files_dir, exist_ok=True)
    output_path = os.path.join(Path(csv_files_dir),f"{filename}.csv")
    df.to_csv(output_path, index=False)
    print(f"file saved to {output_path}")

    segment_save_dir = Path(figures_dir) / "segments_animation_filtered" / motion_name
    os.makedirs(segment_save_dir, exist_ok=True)

    # use previous segmentation for this position csv file
    segments, boundaries = visualize_motion_with_segmentation(filename,
                                                              csv_file_path = output_path,
                                                              wrist_joints=['LWrist', 'RWrist'],
                                                              ankle_joints=['LAnkle', 'RAnkle'],
                                                              save_dir=segment_save_dir)

    for i, boundary in enumerate(boundaries):
        viewer = Viewer(use_reloader=True, xy_size=5, framerate=30)
        Viewer.run = run_patched
        viewer.add_skeleton(pos[boundary[0] or 0:boundary[1], :, :], parents)
        viewer.add_floor()
        # viewer.run()

        print("Generating GIF... this may take a moment.")
        frames = []
        for j in range(viewer.max_frames):
            fig = viewer._create_figure(frame=j)
            img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
            frames.append(imageio.imread(img_bytes))
            if j % 10 == 0:
                print(f"Processed frame {j}/{viewer.max_frames}")
        imageio.mimsave(segment_save_dir/f'seg{i}_{filename}.gif', frames, fps = 30, loop=0)
        print("Saved animation")

## plot trajectories for all positions, axis-angle rep, quaternion rep

    plt.plot(pos[:, 2, 0])
    plt.plot(pos[:, 2, 1])
    plt.plot(pos[:, 2, 2])
    plt.title("Position representation")
    plt.legend(["x","y","z"])
    plt.xlabel("Frame")
    plt.savefig(os.path.join(segment_save_dir, "position_trajectory.png"))
    plt.close()

    # ===== Convert to Quaternions =====
    T, N, _, _ = rotmats.shape
    quaternions = np.zeros((T, N, 4))  # (x, y, z, w) format
    for t in range(T):
        for j in range(N):
            rot = R.from_matrix(rotmats[t, j])
            quaternions[t, j] = rot.as_quat()  # Returns (x, y, z, w)

    print(f"Quaternions shape: {quaternions.shape}")  # (T, N, 4)
    plt.plot(quaternions[:, 2, 0])
    plt.plot(quaternions[:, 2, 1])
    plt.plot(quaternions[:, 2, 2])
    plt.plot(quaternions[:, 2, 3])
    plt.title("Quaternions representation")
    plt.legend(["x","y","z","w"])
    plt.xlabel("Frame")
    plt.savefig(os.path.join(segment_save_dir,"quaternions_trajectory.png"))
    plt.close()

    # ===== Convert to Axis-Angle =====
    axis_angles = np.zeros((T, N, 3))  # Scaled axis-angle representation
    for t in range(T):
        for j in range(N):
            rot = R.from_matrix(rotmats[t, j])
            axis_angles[t, j] = rot.as_rotvec()  # Returns axis * angle

    print(f"Axis-angle shape: {axis_angles.shape}")  # (T, N, 3)
    plt.plot(axis_angles[:, 2, 1])
    plt.plot(axis_angles[:, 2, 1])
    plt.plot(axis_angles[:, 2, 2])
    plt.title("Axis-angle representation")
    plt.legend(["x","y","z"])
    plt.xlabel("Frame")
    plt.savefig(os.path.join(segment_save_dir,"axis_angles_trajectory.png"))
    plt.close()

    # boundary = boundaries[0]
    # # it doeasnt work when boundary[0] == 0 so we put or 0  to make it true
    # viewer.add_skeleton(pos[boundary[0] or 0 :boundary[1],:,:], parents)
    # # add additional info using add_sphere(...) and/or add_line(...), examples:
    # # viewer.add_sphere(sphere_pos, color="green")
    # # viewer.add_line(start_pos, end_pos, color="green")
    # viewer.add_floor()
    # viewer.run()



if __name__ == "__main__":
    main()
