from tkinter.messagebox import showerror
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import imageio.v2 as imageio
from matplotlib.animation import FuncAnimation

# from mp_movement_classifier.utils.utils import calculate_angular_velocity_quat,segment_motion_csv
from mp_movement_classifier.utils import config
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import from_root_positions, fk
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import webbrowser
import os
from pymotion.render.viewer import Viewer


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

def extract_all_motions_csv_generate_animation(path,finished_motions,id_to_motion_name,joint_names,figures_dir):

    for bvh_reference in path.glob("*.bvh"):
        filename = bvh_reference.stem
        # repeat the process for those are not in finished
        if filename not in finished_motions:
            print(filename)
            motion_id_str = filename.split('_')[-1]
            motion_name = id_to_motion_name.get(int(motion_id_str))
            print(motion_name)

            bvh = BVH()
            bvh.load(bvh_reference)  # load euler angle rep from bvh data
            local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
            global_positions = local_positions[:, 0, :]  # root joint
            pos, rotmats = fk(local_rotations, global_positions, offsets, parents)

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
            output_path = os.path.join(Path(csv_files_dir), f"{filename}.csv")
            df.to_csv(output_path, index=False)
            print(f"file saved to {output_path}")

            animations_save_dir = Path(figures_dir) / "all_motions_animation" / motion_name
            os.makedirs(animations_save_dir, exist_ok=True)

            viewer = Viewer(use_reloader=True, xy_size=5, framerate=30)
            Viewer.run = run_patched
            viewer.add_skeleton(pos[:, :, :], parents)
            viewer.add_floor()
            print("Generating GIF... this may take a moment.")
            frames = []
            for j in range(viewer.max_frames):
                fig = viewer._create_figure(frame=j)
                img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
                frames.append(imageio.imread(img_bytes))
                if j % 20 == 0:
                    print(f"Processed frame {j}/{viewer.max_frames}")
            imageio.mimsave(animations_save_dir / f'{filename}.gif', frames, fps=30, loop=0)
            print("Saved animation")

def extract_exponential_maps_csv(path,finished_motions,id_to_motion_name,joint_names,save_dir):

    for bvh_reference in path.glob("*.bvh"):
        filename = bvh_reference.stem
        # repeat the process for those are not in finished
        if filename in finished_motions:
            print(filename)
            motion_id_str = filename.split('_')[-1]
            motion_name = id_to_motion_name.get(int(motion_id_str))
            print(motion_name)

            bvh = BVH()
            bvh.load(bvh_reference)  # load euler angle rep from bvh data
            local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
            global_positions = local_positions[:, 0, :]  # root joint
            pos, rotmats = fk(local_rotations, global_positions, offsets, parents)

            T, N, _, _ = rotmats.shape

            # Convert to Quaternions
            quaternions = np.zeros((T, N, 4))  # (x, y, z, w) format
            for t in range(T):
                for j in range(N):
                    rot = R.from_matrix(rotmats[t, j])
                    quaternions[t, j] = rot.as_quat()  # Returns (x, y, z, w)

            # enforce continuity BEFORE axis-angle conversion
            quaternions = make_quaternions_continuous(quaternions)

            axis_angles = np.zeros((T, N, 3))
            for t in range(T):
                for j in range(N):
                    rot = R.from_quat(quaternions[t, j])  # from continuous quat
                    axis_angles[t, j] = rot.as_rotvec()
            # Optional: unwrap any remaining jumps
            for j in range(N):
                axis_angles[:, j, :] = unwrap_axis_angle(axis_angles[:, j, :])


            columns = []
            data = []
            for joint_idx, joint_name in enumerate(joint_names):
                columns.append(joint_name + "_x")
                columns.append(joint_name + "_y")
                columns.append(joint_name + "_z")

            for frame, pose in enumerate(axis_angles):
                data_frame = []
                for joint_idx, joint_name in enumerate(joint_names):
                    data_frame.extend(pose[joint_idx, :])
                data.append(data_frame)

            df = pd.DataFrame(data, columns=columns)
            csv_files_dir = Path(save_dir)
            os.makedirs(csv_files_dir, exist_ok=True)
            output_path = os.path.join(csv_files_dir, f"{filename}.csv")
            df.to_csv(output_path, index=False)
            print(f"file saved to {output_path}")


def visualize_segment_boundaries(motions_to_visualize, data, id_to_motion_name, folder_path, figures_dir, frame_time):

    for motion in motions_to_visualize:
        boundaries = data[motion]
        filename = motion + ".csv"
        motion_id_str = motion.split('_')[-1]
        motion_name = id_to_motion_name.get(int(motion_id_str))
        csv_path = os.path.join(Path(folder_path), filename)
        motion_df = pd.read_csv(csv_path)
        # channels = df.columns
        time_vector = np.arange(motion_df.shape[0]) * frame_time

        target_joints=["LWrist","LKnee","LElbow","LAnkle","Neck","LShoulder"]

        # Create plots
        fig, axes = plt.subplots(len(target_joints), 1, figsize=(16, 5 * len(target_joints)))
        if len(target_joints) == 1:
            axes = [axes]
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

        # Iterate through target joints
        for i, joint_name in enumerate(target_joints):
            columns = [col for col in motion_df.columns if col.startswith(joint_name)]

            ax = axes[i]
            ax.set_title(f'{motion_name}, {joint_name} Joint rep with Motion Segments',
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
            segment_colors = plt.cm.viridis(np.linspace(0, 1, len(boundaries)))
            j = 0
            for boundary in boundaries:
                start_time = time_vector[boundary[0]]
                end_time = time_vector[boundary[1]]
                ax.axvline(x=start_time, color='r', linestyle='--', alpha=0.7)
                ax.axvline(x=end_time, color='r', linestyle='--', alpha=0.7)
                ax.axvspan(start_time, end_time, color=segment_colors[j], alpha=0.2,
                           label=f'Segment {j + 1}')
                j=j+1

            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Angle (degrees)', fontsize=12)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, time_vector[-1])

        plt.tight_layout()

        seg_dir = os.path.join(figures_dir, "segmentation_boundaries")
        os.makedirs(seg_dir, exist_ok=True)
        plt.savefig(os.path.join(seg_dir, f"{motion}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()


def set_camera_view(fig, view='right'):
    # Define camera positions
    # eye: position of camera (x, y, z)
    # center: what the camera looks at (usually origin)
    # up: which direction is "up" (usually z-axis)

    camera_presets = {
        'left': dict(
            eye = dict(x=-1.25, y=1.25, z=1.25),
            center = dict(x=0, y=0, z=0),
            up = dict(x=0, y=0, z=1)
        ),

        'right': dict(
            eye=dict(x=1.25, y=-1.25, z=1.25),  # Camera on right side
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
    }

    if view.lower() not in camera_presets:
        print(f"Unknown camera view '{view}', using 'front'")
        view = 'front'

    camera = camera_presets[view.lower()]
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            aspectmode='data',  # Maintain aspect ratio
        )
    )

    return fig


def create_animation(motion_file_stem="subject_42_motion_02", camera_view='front'):
    animations_save_dir = os.path.join("./../../results/segmentation_analysis/segments_animation")
    Path(animations_save_dir).mkdir(exist_ok=True)

    path = Path("../../data/bvh_files")
    MAPPING_FILE = "../../data/common_motion_mapping.json"
    with open(MAPPING_FILE, 'r') as f:
        data = json.load(f)
        motion_mapping = data["mapping"]
    id_to_motion_name = {id_val: motion_name for motion_name, id_val in motion_mapping.items()}

    bvh_reference = os.path.join(path, motion_file_stem + ".bvh")
    motion_id_str = motion_file_stem.split('_')[-1]
    motion_name = id_to_motion_name.get(int(motion_id_str))

    # Load BVH and compute FK
    bvh = BVH()
    bvh.load(bvh_reference)
    local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
    global_positions = local_positions[:, 0, :]
    pos, rotmats = fk(local_rotations, global_positions, offsets, parents)

    viewer = Viewer(use_reloader=True, xy_size=5, framerate=30)
    Viewer.run = run_patched
    viewer.add_skeleton(pos[:, :, :], parents)
    viewer.add_floor()
    # viewer.run()

    print("Generating GIF... this may take a moment.")
    frames = []
    for j in range(viewer.max_frames):
        fig = viewer._create_figure(frame=j)
        fig = set_camera_view(fig, camera_view)

        img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
        frames.append(imageio.imread(img_bytes))
        if j % 20 == 0:
            print(f"Processed frame {j}/{viewer.max_frames}")

    output_filename = f'{motion_file_stem}_{camera_view}.gif'
    imageio.mimsave(Path(animations_save_dir) / output_filename, frames, fps=30, loop=0)
    print(f"Saved animation with {camera_view} view")

def make_quaternions_continuous(quats):
    """
    Ensure quaternion continuity across frames.
    quats: shape (T, 4) or (T, N_joints, 4)
    """
    result = quats.copy()
    if result.ndim == 3:
        for j in range(result.shape[1]):
            for t in range(1, result.shape[0]):
                if np.dot(result[t, j], result[t-1, j]) < 0:
                    result[t, j] = -result[t, j]
    elif result.ndim == 2:
        for t in range(1, result.shape[0]):
            if np.dot(result[t], result[t-1]) < 0:
                result[t] = -result[t]
    return result

def unwrap_axis_angle(rotvecs):
    """
    Unwrap axis-angle vectors to remove ±π discontinuities.
    rotvecs: shape (T, 3) for one joint
    """
    result = rotvecs.copy()
    for t in range(1, len(result)):
        diff = result[t] - result[t-1]
        angle_diff = np.linalg.norm(diff)
        if angle_diff > np.pi:
            # Large jump — the rotation went "the other way around"
            angle_t = np.linalg.norm(result[t])
            if angle_t > 1e-6:
                axis_t = result[t] / angle_t
                # Adjust by 2π in the direction that reduces the jump
                new_angle = angle_t - 2 * np.pi
                candidate = axis_t * new_angle
                if np.linalg.norm(candidate - result[t-1]) < angle_diff:
                    result[t] = candidate
    return result

def plot_diff_representations(motion_file_stem,joint_names):
    save_dir = os.path.join("./../../results/segmentation_analysis/diff_representations")
    Path(save_dir).mkdir(exist_ok=True)
    joints_to_plot = ["LWrist", "LKnee", "LElbow", "LAnkle", "Neck", "LShoulder"]
    joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

    # Load motion data
    path = Path("../../data/bvh_files")
    MAPPING_FILE = "../../data/common_motion_mapping.json"
    with open(MAPPING_FILE, 'r') as f:
        data = json.load(f)
        motion_mapping = data["mapping"]
    id_to_motion_name = {id_val: motion_name for motion_name, id_val in motion_mapping.items()}

    bvh_reference = os.path.join(path, motion_file_stem + ".bvh")
    motion_id_str = motion_file_stem.split('_')[-1]
    motion_name = id_to_motion_name.get(int(motion_id_str))

    bvh = BVH()
    bvh.load(bvh_reference)
    local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
    global_positions = local_positions[:, 0, :]
    pos, rotmats = fk(local_rotations, global_positions, offsets, parents)

    # Step 2: Convert to different representations
    T, N, _, _ = rotmats.shape

    # Convert to Quaternions
    quaternions = np.zeros((T, N, 4))  # (x, y, z, w) format
    for t in range(T):
        for j in range(N):
            rot = R.from_matrix(rotmats[t, j])
            quaternions[t, j] = rot.as_quat()  # Returns (x, y, z, w)

    # enforce continuity BEFORE axis-angle conversion
    quaternions = make_quaternions_continuous(quaternions)

    # Convert to Axis-Angle
    # axis_angles = np.zeros((T, N, 3))  # Scaled axis-angle representation
    # for t in range(T):
    #     for j in range(N):
    #         rot = R.from_matrix(rotmats[t, j])
    #         axis_angles[t, j] = rot.as_rotvec()  # Returns axis * angle

    axis_angles = np.zeros((T, N, 3))
    for t in range(T):
        for j in range(N):
            rot = R.from_quat(quaternions[t, j])  # from continuous quat
            axis_angles[t, j] = rot.as_rotvec()
    # Step 3: Plot each representation for each joint
    num_joints = len(joints_to_plot)

    # Optional: unwrap any remaining jumps
    for j in range(N):
        axis_angles[:, j, :] = unwrap_axis_angle(axis_angles[:, j, :])

    # ========== POSITION REPRESENTATION ==========
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 4 * num_joints))
    if num_joints == 1:
        axes = [axes]

    for idx, joint_name in enumerate(joints_to_plot):
        if joint_name not in joint_name_to_idx:
            print(f"Warning: Joint {joint_name} not found in joint_names")
            continue

        joint_idx = joint_name_to_idx[joint_name]
        ax = axes[idx]

        ax.plot(pos[:, joint_idx, 0], label='x', linewidth=1.5)
        ax.plot(pos[:, joint_idx, 1], label='y', linewidth=1.5)
        ax.plot(pos[:, joint_idx, 2], label='z', linewidth=1.5)
        ax.set_title(f"Position - {joint_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Position")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{motion_file_stem}_position_all_joints.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {motion_file_stem}_position_all_joints.png")

    # ========== QUATERNION REPRESENTATION ==========
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 4 * num_joints))
    if num_joints == 1:
        axes = [axes]

    for idx, joint_name in enumerate(joints_to_plot):
        if joint_name not in joint_name_to_idx:
            continue

        joint_idx = joint_name_to_idx[joint_name]
        ax = axes[idx]

        ax.plot(quaternions[:, joint_idx, 0], label='x', linewidth=1.5)
        ax.plot(quaternions[:, joint_idx, 1], label='y', linewidth=1.5)
        ax.plot(quaternions[:, joint_idx, 2], label='z', linewidth=1.5)
        ax.plot(quaternions[:, joint_idx, 3], label='w', linewidth=1.5)
        ax.set_title(f"Quaternion - {joint_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Quaternion Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{motion_file_stem}_quaternions_all_joints.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {motion_file_stem}_quaternions_all_joints.png")

    # ========== AXIS-ANGLE REPRESENTATION ==========
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 4 * num_joints))
    if num_joints == 1:
        axes = [axes]

    for idx, joint_name in enumerate(joints_to_plot):
        if joint_name not in joint_name_to_idx:
            continue

        joint_idx = joint_name_to_idx[joint_name]
        ax = axes[idx]

        ax.plot(axis_angles[:, joint_idx, 0], label='x', linewidth=1.5)
        ax.plot(axis_angles[:, joint_idx, 1], label='y', linewidth=1.5)
        ax.plot(axis_angles[:, joint_idx, 2], label='z', linewidth=1.5)
        ax.set_title(f"Axis-Angle - {joint_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Rotation (radians)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{motion_file_stem}_axis_angles_all_joints.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {motion_file_stem}_axis_angles_all_joints.png")

    print(f"\nAll plots saved to: {save_dir}")


def main():
    joint_names = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Thorax', 'Neck',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
    ]
    figures_dir = os.path.join("./../../results/segmentation_analysis")
    Path(figures_dir).mkdir(exist_ok=True)

    path = Path("../../data/bvh_files")
    MAPPING_FILE = "../../data/common_motion_mapping.json"
    with open(MAPPING_FILE, 'r') as f:
        data = json.load(f)
        motion_mapping = data["mapping"]
    id_to_motion_name = {id_val: motion_name for motion_name, id_val in motion_mapping.items()}


    motions_to_visualize = [
        "subject_16_motion_02", "subject_9_motion_05","subject_59_motion_18","subject_12_motion_17","subject_2_motion_11",
        "subject_28_motion_09","subject_23_motion_08","subject_21_motion_03", "subject_32_motion_00","subject_70_motion_06",
        "subject_15_motion_01","subject_60_motion_12","subject_13_motion_07","subject_17_motion_13","subject_75_motion_10",
        "subject_19_motion_14"
    ]
    Segments_index = "../../results/segments_index.json"
    with open(Segments_index, 'r') as f:
        data = json.load(f)

    folder_path = "../../data/pymotion_exponential_csv_files"
    frame_time = 1/30

    # finished_motions = []
    with open('files.txt', 'r') as f:
        finished_motions = [line.strip() for line in f if line.strip()]
        print(finished_motions)
    # extract_all_motions_csv_generate_animation(path, finished_motions, id_to_motion_name, joint_names, figures_dir)

    # visualize_segment_boundaries(motions_to_visualize, data, id_to_motion_name, folder_path, figures_dir, frame_time)

    # rotate camera view to right so that the front of subject can be seen
    # create_animation(motion_file_stem="subject_52_motion_00", camera_view='right')

    # plot different represenation of motion trajectory
    # plot_diff_representations(  "subject_16_motion_02", joint_names)
    extract_exponential_maps_csv(path, finished_motions, id_to_motion_name, joint_names, save_dir=folder_path)

if __name__ == "__main__":
    main()


