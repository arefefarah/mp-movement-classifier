from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import from_root_positions, fk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from mp_movement_classifier.utils.h36m_csv_converter import H36MConverter


path = Path("../../data/MMpose/df_files_3d")

output_dir = Path("../../data/pymotion_quat_csv_files")
output_dir.mkdir(exist_ok=True)


converter = H36MConverter()

for csv_file in path.glob("*.csv"):
    bvh_file = Path("../../data/bvh_files") / csv_file.name.replace(".csv", ".bvh")
    out_file = output_dir / csv_file.name

    bvh = BVH()
    bvh.load(bvh_file)
    local_rotations, local_positions, parents,offsets, end_sites, _= bvh.get_data()
    # joint_names = bvh.data['names']
    #format names to be compatible with previous calculations
    joint_names = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Thorax', 'Neck',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
    ]
    # print(f"Number of joints: {len(parents)}")
    mmpose_positions = converter.convert_csv_to_numpy(csv_file)  # Shape: (T, N, 3)

    # Convert positions to quaternion rotations
    quaternion_rotations = from_root_positions(mmpose_positions, parents, offsets)
    # Returns shape: (T, N, 4) - quaternions in (x, y, z, w) format

    # Reorder to (w, x, y, z) format to be compatible with quaternion angular speed calculation
    quaternion_rotations_wxyz = np.roll(quaternion_rotations, shift=1, axis=-1)
    # New shape: (T, N, 4) in (w, x, y, z) format

    columns = []
    for joint_idx, joint_name in enumerate(joint_names):

        columns.append(joint_name + "_w")
        columns.append(joint_name +"_x")
        columns.append(joint_name +"_y")
        columns.append(joint_name +"_z")

    #reshape
    quaternion_data = quaternion_rotations.reshape(quaternion_rotations.shape[0], -1)
    df = pd.DataFrame(quaternion_data, columns=columns)
    df.to_csv(out_file, index=False)
    print(f"file saved to {out_file}")


def visualize_trajectory(quaternion_rotations, joint_names):
    joint_indices = [11, 10]
    joint_names_toplot = [joint_names[i] for i in joint_indices]
    # Change these to your joints of interest

    frames = np.arange(quaternion_rotations.shape[0])

    component_names = ['w','x', 'y', 'z']
    colors = ['red', 'green', 'blue', 'orange']

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle('All Quaternion Components per Joint', fontsize=16, fontweight='bold')

    for joint_idx, (joint_id, joint_name) in enumerate(zip(joint_indices, joint_names_toplot)):
        ax = axes[joint_idx]
        joint_quats = quaternion_rotations[:, joint_id, :]

        # Plot all 4 components
        for comp_idx, (comp_name, color) in enumerate(zip(component_names, colors)):
            ax.plot(frames, joint_quats[:, comp_idx],
                    color=color, linewidth=1.5, label=f'{comp_name}', alpha=0.8)

        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('Quaternion Value', fontsize=11)
        ax.set_title(f'{joint_name} (Index: {joint_id})', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


# visualize_trajectory(quaternion_rotations, joint_names)