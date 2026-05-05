import numpy as np
import re
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import imageio.v2 as imageio
from matplotlib.animation import FuncAnimation

from mp_movement_classifier.tmp_extraction.weights_analysis import reconstruct_segments_with_avg_weights
from mp_movement_classifier.benchmark_analysis.legendre_extraction import generate_legendre_basis
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



def reconstruct_from_model(method_name , model_dir,joint_names,motion_name,desired_length,
                           root_translation, offsets, parents):
    if method_name == "legandre_position" or method_name == "legandre_exponential":
        model_dir = os.path.join(model_dir, "legandre_analysis")
        avg_weights_path = os.path.join(model_dir, "averaged_weights")
    else:
        avg_weights_path = os.path.join(model_dir, "averaged_weights")

    file_path = f"{avg_weights_path}/avg_weights_{motion_name}.npz"
    avg_weights_data = np.load(file_path)
    avg_weights = avg_weights_data['mean_weights']
    avg_weights_data.close()
    if method_name == "tmp_position":
        reconstructed = reconstruct_segments_with_avg_weights(
            model_path=os.path.join(model_dir, f"mp_model_5_PC_tpoints_30"),
            avg_weights=avg_weights,
            segment_length=desired_length
        )
        pos = reconstructed.T.reshape(-1, len(joint_names), 3)
        return reconstructed, pos

    elif method_name == "legandre_position":
        avg_coefficients = avg_weights
        t = np.linspace(0, 1, desired_length)
        max_degree = 1
        basis = generate_legendre_basis(max_degree, t)  # (time_steps, max_degree+1)
        reconstructed = avg_coefficients @ basis.T
        pos = reconstructed.T.reshape(-1, len(joint_names), 3)
        return reconstructed, pos

    elif method_name == "legandre_exponential":
        avg_coefficients = avg_weights
        t = np.linspace(0, 1, desired_length)
        max_degree = 1
        basis = generate_legendre_basis(max_degree, t)  # (time_steps, max_degree+1)
        reconstructed = avg_coefficients @ basis.T

    elif method_name == "tmp_exponential_map":
        # first xonstrcut in exponential  map
        reconstructed = reconstruct_segments_with_avg_weights(
            model_path=os.path.join(model_dir, f"mp_model_5_PC_tpoints_30"),
            avg_weights=avg_weights,
            segment_length=desired_length
        )
    axis_angles = reconstructed.T.reshape(-1, len(joint_names), 3)
    T_recon = axis_angles.shape[0]
    n_joints = axis_angles.shape[1]
    rotmats = np.zeros((T_recon, n_joints, 3, 3))
    for t in range(T_recon):
        for j in range(n_joints):
            angle = np.linalg.norm(axis_angles[t, j])
            if angle < 1e-8:
                rotmats[t, j] = np.eye(3)
            else:
                rotmats[t, j] = R.from_rotvec(axis_angles[t, j]).as_matrix()

    root_pos = root_translation
    offsets = np.array(offsets)  # (n_joints, 3)
    parents = np.array(parents)
    positions = np.zeros((T_recon, n_joints, 3))

    for t in range(T_recon):
        for j in range(n_joints):
            if parents[j] == -1:
                # Root joint
                positions[t, j] = root_pos[t]
            else:
                # Child: parent_position + parent_global_rotation @ bone_offset
                p = parents[j]
                positions[t, j] = positions[t, p] + rotmats[t, p] @ offsets[j]
    return reconstructed,positions

def get_pos_info(filename):
    bvh_reference = f"../../data/bvh_files/{filename}.bvh"
    bvh = BVH()
    bvh.load(bvh_reference)  # load euler angle rep from bvh data
    local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
    global_positions = local_positions[:, 0, :]  # root joint
    pos_original, rotmats = fk(local_rotations, global_positions, offsets, parents)
    print(f"original pos shape: {pos_original.shape}") # pos shape: (177, 16, 3) 177 num of frames, 16 joint, 3 coordinate= 48 channels total
    return global_positions,parents,offsets

def main():
    joint_names = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Thorax', 'Neck',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
    ]
    model_dir = os.path.join("./../../results/tmp_configs", f"new_seg_pymotion_position_mp_model_5_phase_two")
    # model_dir = os.path.join("./../../results/tmp_configs",
    #                          f"new_seg_exponential_mp_model_5_tpoints_30_phase_two")
    output_path = os.path.join(model_dir, "reconstruction")
    Path(output_path).mkdir(exist_ok=True)
    MAPPING_FILE = "../../data/common_motion_mapping.json"

    with open(MAPPING_FILE, 'r') as f:
        data = json.load(f)
        motion_mapping = data["mapping"]
    id_to_motion_name = {id_val: motion_name for motion_name, id_val in motion_mapping.items()}

    motions_to_visualize = [
        "subject_16_motion_02", "subject_9_motion_05", "subject_59_motion_18", "subject_12_motion_17",
        "subject_2_motion_11","subject_28_motion_09", "subject_23_motion_08", "subject_21_motion_03",
        "subject_32_motion_00","subject_70_motion_06","subject_15_motion_01", "subject_60_motion_12",
        "subject_33_motion_00", "subject_13_motion_07","subject_17_motion_13", "subject_75_motion_10",
        "subject_19_motion_14"
    ]
    for filename in motions_to_visualize:

    # filename = f"subject_13_motion_18"
        global_positions, parents, offsets = get_pos_info(filename)
        motion_id_str = filename.split('_')[-1]
        motion_name = id_to_motion_name.get(int(motion_id_str))

        ### here we want to replace pos with recontructed values
        # "legandre_position" ,"legandre_exponential", "tmp_exponential_map" , "tmp_position"
        method_name = "tmp_position"
        reconstructed, pos = reconstruct_from_model(method_name, model_dir, joint_names,motion_name=motion_name,
                                                    desired_length=50,
                                                    root_translation = global_positions,parents = parents,offsets = offsets)

        output = os.path.join(output_path, f"recon_from_{method_name}_segment_motion_{motion_name}.npy")
        np.save(output, reconstructed)
        print(f"saved recon to {output}")
        viewer = Viewer(use_reloader=True, xy_size=5, framerate=30)
        Viewer.run = run_patched
        viewer.add_skeleton(pos, parents)
        viewer.add_floor()
        print(f"Generating GIF for {filename}... this may take a moment.")
        frames = []
        for j in range(viewer.max_frames):
            fig = viewer._create_figure(frame=j)
            img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
            frames.append(imageio.imread(img_bytes))
            if j % 10 == 0:
                print(f"Processed frame {j}/{viewer.max_frames}")

        imageio.mimsave(Path(output_path) / f'{filename}_recon_from_{method_name}.gif', frames, fps = 30, loop=0)
        print("Saved animation")

        plt.plot(pos[:, 2, 0])
        plt.plot(pos[:, 2, 1])
        plt.plot(pos[:, 2, 2])
        plt.title("RKnee joint trajectory")
        plt.legend(["x","y","z"])
        plt.xlabel("Frame")
        plt.savefig(os.path.join(output_path, f"RKnee_{filename}_{method_name}.png"))
        plt.close()


if __name__ == "__main__":
    main()
