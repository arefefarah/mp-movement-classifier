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

from mp_movement_classifier.utils.utils import calculate_angular_velocity_quat
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

# def visualize_motion_with_segmentation(file_name,csv_file_path, wrist_joints , ankle_joints,save_dir):
#
#     motion_df = pd.read_csv(csv_file_path)
#     frame_rate = 30
#     frame_time = 1 / frame_rate
#     segments, boundaries = segment_motion_csv(csv_file_path, data_type= "position",
#                                               wrist_joints = wrist_joints,
#                                               ankle_joints = ankle_joints,
#                                               filtering=False)
#     print(f"len of segments: {len(segments)}")
#     boundary_frames = [boundaries[0][0]] + [b[1] for b in boundaries]
#
#     # Create time vector
#     time_vector = np.arange(motion_df.shape[0]) * frame_time
#
#     target_joints=["LWrist","LKnee","LElbow","LAnkle","Neck","LShoulder"]
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
#         columns = [col for col in motion_df.columns if col.startswith(joint_name)]
#         axis_angle_rep = motion_df[columns]
#
#         ax = axes[i]
#         ax.set_title(f'{joint_name} Joint rep with Motion Segments',
#                      fontsize=16, fontweight='bold')
#
#         # Plot each rotation channel
#         for idx,column in enumerate(columns):
#             color = colors[idx % len(colors)]
#             ax.plot(time_vector,motion_df[column],
#                     color=color,
#                     label=f'{column}',
#                     linewidth=1.5,
#                     alpha=0.7)
#
#         # Plot segment boundaries
#         for boundary in boundary_frames[1:-1]:  # Exclude first and last
#             # ax.axvline(x=time_vector[boundary], color='r', linestyle='--', alpha=0.7)
#             ax.axvline(x=time_vector[int(boundary)], color='r', linestyle='--', alpha=0.7)
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
#     figures_dir = os.path.join(save_dir, "motion_segmentation")
#     os.makedirs(figures_dir, exist_ok=True)
#     plt.savefig(os.path.join(figures_dir, f"{file_name}.png"),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
#
#     return segments,boundaries


def main():
    joint_names = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Thorax', 'Neck',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
    ]
    model_dir = os.path.join("./../../results/tmp_configs", f"new_seg_pymotion_position_mp_model_5_phase_two")

    # model_dir = os.path.join(model_dir, "legandre_analysis")
    output_path = os.path.join(model_dir, "reconstruction")
    Path(output_path).mkdir(exist_ok=True)
    # figures_dir = os.path.join("./../../results/segmentation_analysis")
    # Path(figures_dir).mkdir(exist_ok=True)
    # for i in range(9):
    filename = f"subject_4_motion_05"
    # csv_file = f"../../data/pymotion_quat_csv_files/{filename}.csv"
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
    pos_original, rotmats = fk(local_rotations, global_positions, offsets, parents)
    print(f"pos shape: {pos_original.shape}") # pos shape: (177, 16, 3) 177 num of frames, 16 joint, 3 coordinate= 48 channels total

    ### here we want to replace pos with recontructed values of TMP model
    pos_file = os.path.join(model_dir, "tmp_reconstructed_segment_motion_5.npy")

    tmp_array = np.load(pos_file) #array shape num_frmes, channels(16*3)
    pos = tmp_array.T.reshape(-1, len(joint_names), 3)

    # columns = []
    # data = []
    # for joint_idx, joint_name in enumerate(joint_names):
    #     columns.append(joint_name + "_x")
    #     columns.append(joint_name + "_y")
    #     columns.append(joint_name + "_z")
    #
    # for frame, pose in enumerate(pos):
    #     data_frame = []
    #     for joint_idx, joint_name in enumerate(joint_names):
    #         data_frame.extend(pose[joint_idx, :])
    #     data.append(data_frame)
    #
    # df = pd.DataFrame(data, columns=columns)
    # output_dir = os.path.join(output_path,f"{filename}.csv")
    # df.to_csv(output_dir, index=False)
    # print(f"file saved to {output_dir}")

    # segment_save_dir = Path(output_path) / "segments_animation" / motion_name
    # os.makedirs(segment_save_dir, exist_ok=True)

    # # use previous segmentation for this position csv file
    # segments, boundaries = visualize_motion_with_segmentation(filename,
    #                                                           csv_file_path = output_dir,
    #                                                           wrist_joints=['LWrist', 'RWrist'],
    #                                                           ankle_joints=['LAnkle', 'RAnkle'],
    #                                                           save_dir=segment_save_dir)

    # for i, boundary in enumerate(boundaries):
    viewer = Viewer(use_reloader=True, xy_size=5, framerate=30)
    Viewer.run = run_patched
    viewer.add_skeleton(pos, parents)
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

    # imageio.mimsave(f'{filename}.gif', frames, fps=30, loop=0)
    imageio.mimsave(Path(output_path) / f'{filename}_tmp_recons.gif', frames, fps = 30, loop=0)
    print("Saved animation")
    # viewer.run()

## plot trajectories for all positions, axis-angle rep, quaternion rep

    plt.plot(pos[:, 2, 0])
    plt.plot(pos[:, 2, 1])
    plt.plot(pos[:, 2, 2])
    plt.title("Position representation")
    plt.legend(["x","y","z"])
    plt.xlabel("Frame")
    plt.savefig(os.path.join(output_path, "position_trajectory.png"))
    plt.close()


if __name__ == "__main__":
    main()
