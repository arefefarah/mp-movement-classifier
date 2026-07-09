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
from collections import defaultdict
from pymotion.render.viewer import Viewer

from mp_movement_classifier.utils.utils import process_motion_data


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
            # All fonts ≥ 12 pt to match the single-joint figure and stay
            # legible when scaled down into a manuscript column.
            ax.set_title(f'{motion_name}, {joint_name} Joint rep with Motion Segments',
                         fontsize=20, fontweight='bold')

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

            ax.set_xlabel('Time (seconds)', fontsize=16)
            ax.set_ylabel('Angle (degrees)', fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(fontsize=14, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, time_vector[-1])

        plt.tight_layout()

        seg_dir = os.path.join(figures_dir, "segmentation_boundaries")
        os.makedirs(seg_dir, exist_ok=True)
        plt.savefig(os.path.join(seg_dir, f"{motion}.png"),
                    dpi=300, bbox_inches='tight')
        print(f"Saved: {motion}.png")
        plt.close()

def plot_single_joint_boundaries(data, id_to_motion_name, frame_time, folder_path, figures_dir, motion, joint_name):
    boundaries = data[motion]
    filename = motion + ".csv"
    motion_id_str = motion.split('_')[-1]
    motion_name = id_to_motion_name.get(int(motion_id_str))
    csv_path = os.path.join(Path(folder_path), filename)
    motion_df = pd.read_csv(csv_path)
    time_vector = np.arange(motion_df.shape[0]) * frame_time
    columns = [col for col in motion_df.columns if col.startswith(joint_name)]

    # ── Central font-size config ─────────────────────────────────────────────
    # All values are ≥ 12 pt so the figure remains legible even when scaled
    # down to a paper column. Sized in a single dict so future tweaks are
    # one line.
    FONT = dict(
        title=30,   # "{motion} – {joint} Joint with Motion Segments"
        label=28,   # x/y axis labels
        tick=24,    # x/y tick labels
        legend=18,  # legend body
    )

    # ── Design big so when scaled down in Affinity fonts stay readable ────────
    fig, ax = plt.subplots(figsize=(20, 6))

    ax.set_title(f'{motion_name} – {joint_name} Joint with Motion Segments',
                 fontsize=FONT['title'], fontweight='bold')

    channel_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    for idx, column in enumerate(columns):
        ax.plot(time_vector, motion_df[column],
                color=channel_colors[idx % len(channel_colors)],
                label=column,
                linewidth=2,
                alpha=0.7)

    # ── X-axis cap at 10 seconds ──────────────────────────────────────────────
    x_max = min(time_vector[-1], 10.0)

    segment_colors = plt.cm.viridis(np.linspace(0, 1, len(boundaries)))
    for j, boundary in enumerate(boundaries):
        start_time = time_vector[boundary[0]]
        end_time   = time_vector[boundary[1]]

        # skip segments entirely outside the 10s window
        if start_time > x_max:
            continue

        end_time = min(end_time, x_max)  # clip segment end to 10s

        ax.axvline(x=start_time, color='r', linestyle='--', alpha=0.7)
        ax.axvline(x=end_time,   color='r', linestyle='--', alpha=0.7)
        ax.axvspan(start_time, end_time,
                   color=segment_colors[j], alpha=0.2,
                   label=f'Segment {j + 1}')

    ax.set_xlabel('Time (seconds)', fontsize=FONT['label'])
    ax.set_ylabel('Position (meters)', fontsize=FONT['label'])
    ax.tick_params(axis='both', labelsize=FONT['tick'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max)   # enforces the 10s cap on the plot

    ax.legend(fontsize=FONT['legend'],
              loc='upper left',
              bbox_to_anchor=(0, -0.15),   # below the plot instead of beside it
              borderaxespad=0,
              ncol=len(columns) + len(boundaries))

    plt.tight_layout()

    seg_dir = os.path.join(figures_dir, "segmentation_boundaries")
    os.makedirs(seg_dir, exist_ok=True)
    # Save both SVG and PNG so the figure can drop directly into the
    # manuscript without an extra conversion step. ``bbox_inches='tight'``
    # crops both consistently to their actual content.
    svg_path = os.path.join(seg_dir, f"{motion}_{joint_name}.svg")
    png_path = os.path.join(seg_dir, f"{motion}_{joint_name}.png")
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")
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
            aspectmode='data',  # Maintains the true physical proportions
            # these 3 lines are for removing the checkborads but maintain the scale of skeleton
            xaxis=dict(range=[-0.5, 0.5], visible=False),
            yaxis=dict(range=[-0.5, 0.5], visible=False),
            # for 2 special movment(crawling and legged sitting) we need larger space
            # xaxis=dict(range=[-2, 2], visible=False),
            # yaxis=dict(range=[-2, 2], visible=False),
            zaxis=dict(range=[-0.5, 2], visible=False)
        )
    )

    return fig


def create_animation(motion_file_stem="subject_42_motion_02", camera_view='front'):
    animations_save_dir = os.path.join("./../../results/segmentation_analysis/white_back_segments_animation")
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
    # comment out add_floor for for removing the checkborads
    # viewer.add_floor()

    # viewer.run()

    print("Generating GIF... this may take a moment.")
    frames = []
    # for j in range(viewer.max_frames):
    for j in range(80):
        fig = viewer._create_figure(frame=j)
        fig = set_camera_view(fig, camera_view)

        img_bytes = fig.to_image(format="png")
        frames.append(imageio.imread(img_bytes))
        if j % 20 == 0:
            print(f"Processed frame {j}/{viewer.max_frames}")

    output_filename = f'{motion_file_stem}_{camera_view}.gif'
    imageio.mimsave(Path(animations_save_dir) / output_filename, frames, fps=30, loop=0)
    print(f"Saved animation {output_filename} with {camera_view} view")

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


def plot_coordinate_trajectories(
        motions,
        boundaries_json,
        id_to_motion_name,
        folder_path,
        figures_dir,
        frame_time,
        joint_names,
        segment_idx=0,
):
    """
    Dataset-overview plot: joint-position coordinate trajectories for ONE
    subject performing several different motion categories.

    Layout: a grid with one row per motion and three columns (X, Y, Z). Each
    cell overlays all 16 joint trajectories for that (motion, coordinate),
    one colour per joint, with a single shared legend below.

    Parameters
    ----------
    motions : list[str]
    boundaries_json : dict
        Parsed segments_index.json (file stem -> [[start, end], ...]).
        Each motion is cropped to one movement cycle (``segment_idx``).
    segment_idx : int
        Which segment (movement cycle) of each recording to show.
    """
    if isinstance(motions, str):
        motions = [motions]

    axis_labels = ['x', 'y', 'z']
    colors = plt.cm.tab20(np.linspace(0, 1, len(joint_names)))
    colors = ['b','r']
    FONT = dict(coltitle=18, rowlabel=15, label=15, tick=12, legend=12, suptitle=20)

    n_rows = len(motions)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(15, max(2.6 * n_rows, 3.2)),
        squeeze=False,
    )

    for r, motion in enumerate(motions):
        filename = motion + ".csv"
        motion_id_str = motion.split('_')[-1]
        motion_name = id_to_motion_name.get(int(motion_id_str), motion_id_str)
        csv_path = os.path.join(Path(folder_path), filename)
        motion_df = pd.read_csv(csv_path)

        # Crop to a single movement cycle when boundaries are available.
        boundaries = boundaries_json.get(motion)
        if boundaries and segment_idx < len(boundaries):
            b = boundaries[segment_idx]
            seg_df = motion_df.iloc[b[0]:b[1], :].reset_index(drop=True)
        else:
            seg_df = motion_df

        n_frames = seg_df.shape[0]
        time_vector = np.arange(n_frames) * frame_time

        for c in range(3):
            ax = axes[r][c]
            for j, joint_name in enumerate(joint_names):
                col = f"{joint_name}_{axis_labels[c]}"
                if col not in seg_df.columns:
                    continue
                ax.plot(time_vector, seg_df[col],
                        color=colors[j], linewidth=1.2, alpha=0.85,
                        label=joint_name)
            ax.tick_params(axis='both', labelsize=FONT['tick'])
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, time_vector[-1] if n_frames > 1 else 1)

            # Column titles only on the top row.
            if r == 0:
                ax.set_title(f'{axis_labels[c].upper()} position (m)',
                             fontsize=FONT['coltitle'], fontweight='bold')
            # Time axis label only on the bottom row.
            if r == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=FONT['label'])

        # Motion name as the row label on the leftmost cell.
        axes[r][0].set_ylabel(motion_name, fontsize=FONT['rowlabel'],
                              fontweight='bold')

    fig.suptitle('Right and Left wrist Joint position trajectories across motions',
                 fontsize=FONT['suptitle'], fontweight='bold', y=0.998)

    # Single shared legend below the grid.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=8, fontsize=FONT['legend'], frameon=True, framealpha=0.9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    traj_dir = os.path.join(figures_dir, "coordinate_trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    out_png = os.path.join(traj_dir, "subjects_trajectories_across_motions.png")
    out_svg = os.path.join(traj_dir, "subjects_trajectories_across_motions.svg")
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_svg, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")


def plot_segment_duration_barplot(
        folder_path,
        id_to_motion_name,
        figures_dir,
        frame_time,
        data_type="position",
):
    """
    Dataset-overview plot: segment temporal length per motion category.

    Loads every segment across all subjects (via process_motion_data),
    converts each segment length to seconds, and draws one bar per motion
    showing the mean duration with an error bar (± std across segments of
    that motion). Bars are sorted by mean duration so the figure reads as a
    gradient, and the per-motion segment count is annotated above each bar.
    """
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=folder_path, data_type=data_type, filtering=False,
    )

    # Group segment durations (seconds) by motion id.
    durations = defaultdict(list)
    for seg, mid in zip(processed_segments, segment_motion_ids):
        # seg shape is [signals, time]; time dimension = number of frames.
        durations[int(mid)].append(seg.shape[1] * frame_time)

    # Build sorted (by mean duration) arrays for plotting.
    stats = []
    for mid, durs in durations.items():
        durs = np.asarray(durs, dtype=float)
        stats.append((
            id_to_motion_name.get(mid, str(mid)),
            durs.mean(),
            durs.std(),
            len(durs),
        ))
    stats.sort(key=lambda t: t[1])  # ascending by mean duration
    names = [s[0] for s in stats]
    means = np.array([s[1] for s in stats])
    stds = np.array([s[2] for s in stats])
    counts = [s[3] for s in stats]

    FONT = dict(title=18, label=15, tick=13, annot=11)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4,
           color='#4C72B0', edgecolor='black', linewidth=0.6, alpha=0.85)

    # Annotate the number of segments above each bar.
    for xi, m, s, n in zip(x, means, stds, counts):
        ax.text(xi, m + s + 0.02 * means.max(), f"n={n}",
                ha='center', va='bottom', fontsize=FONT['annot'], color='0.25')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=FONT['tick'])
    ax.set_ylabel('Segment duration (s)', fontsize=FONT['label'], fontweight='bold')
    ax.set_xlabel('Motion category', fontsize=FONT['label'], fontweight='bold')
    ax.set_title('Segment duration per motion category (mean ± std)',
                 fontsize=FONT['title'], fontweight='bold')
    ax.tick_params(axis='y', labelsize=FONT['tick'])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_png = os.path.join(figures_dir, "segment_duration_per_motion.png")
    out_svg = os.path.join(figures_dir, "segment_duration_per_motion.svg")
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_svg, bbox_inches='tight', facecolor='white')
    plt.close()

    total = sum(counts)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")
    print(f"  ({total} segments across {len(names)} motions; "
          f"overall mean = {means.mean():.2f} s)")


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
        "subject_9_motion_02",
        # "subject_9_motion_05",
        # "subject_25_motion_14","subject_34_motion_12","subject_15_motion_07","subject_52_motion_01"
        # "subject_4_motion_05","subject_1_motion_03"  #crawling and cross leg sitting
        # "subject_23_motion_13", "subject_8_motion_09","subject_5_motion_02","subject_5_motion_06",
        # "subject_1_motion_03","subject_25_motion_14","subject_5_motion_12","subject_18_motion_08",
        # "subject_8_motion_07","subject_33_motion_18","subject_1_motion_01","subject_4_motion_05",
        # "subject_4_motion_17","subject_13_motion_11","subject_9_motion_10","subject_21_motion_00",
    ]
    Segments_index = "../../data/segments_index.json"
    with open(Segments_index, 'r') as f:
        data = json.load(f)

    folder_path = "../../data/pymotion_position_csv_files"
    frame_time = 1/30

    # finished_motions = []
    # with open('files.txt', 'r') as f:
    #     finished_motions = [line.strip() for line in f if line.strip()]
    #     print(finished_motions)
    # extract_all_motions_csv_generate_animation(path, finished_motions, id_to_motion_name, joint_names, figures_dir)

    # visualize_segment_boundaries(motions_to_visualize, data, id_to_motion_name, folder_path, figures_dir, frame_time)

    # plot segments for figure in paper
    # plot_single_joint_boundaries(
    #     data,id_to_motion_name, frame_time, folder_path, figures_dir,
    #     motion=motions_to_visualize[1],
    #     joint_name="LAnkle",
    # )

    # ── Dataset-overview plot 1: coordinate trajectories for ONE subject
    #    across several different motions. One figure, grid = rows (motions)
    #    × cols (X/Y/Z), all 16 joints overlaid per cell. List the motion
    #    files for a single subject.
    subject_motions = [
        "subject_16_motion_02", "subject_17_motion_05","subject_13_motion_18","subject_12_motion_17",
        "subject_13_motion_11","subject_12_motion_09","subject_31_motion_08","subject_21_motion_03",
        "subject_32_motion_00","subject_12_motion_06","subject_15_motion_01", "subject_10_motion_12",
        "subject_13_motion_07","subject_12_motion_13","subject_13_motion_10","subject_12_motion_14",
    ]
    plot_coordinate_trajectories(
        motions=subject_motions,
        boundaries_json=data,
        id_to_motion_name=id_to_motion_name,
        folder_path=folder_path,
        figures_dir=figures_dir,
        frame_time=frame_time,
        joint_names=['LWrist','RWrist'], #joint_names
        segment_idx=0,
    )

    # ── Dataset-overview plot 2: segment duration per motion (one bar per
    #    motion, mean ± std across all segments of that motion).
    # plot_segment_duration_barplot(
    #     folder_path=folder_path,
    #     id_to_motion_name=id_to_motion_name,
    #     figures_dir=figures_dir,
    #     frame_time=frame_time,
    # )


    # rotate camera view to right so that the front of subject can be seen
    # for motion in motions_to_visualize:
    #     create_animation(motion_file_stem=motion, camera_view='right')

    # plot different represenation of motion trajectory
    # plot_diff_representations(  "subject_4_motion_05", joint_names)
    # extract_exponential_maps_csv(path, finished_motions, id_to_motion_name, joint_names, save_dir=folder_path)

if __name__ == "__main__":
    main()


