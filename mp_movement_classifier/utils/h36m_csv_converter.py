import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
# from bvh_skeleton import h36m_skeleton
from bvh_converter import h36m_skeleton



class H36MConverter:
    def __init__(self):
        # skeleton structure
        self.joint_names = [
            'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
            'Spine', 'Thorax', 'Neck','Head',
            'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
        ]

    def convert_csv_to_numpy(self, csv_path):
        """Convert CSV file to numpy array with correct joint ordering"""
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Get number of frames
        n_frames = len(df['frame_id'].unique())
        n_joints = len(self.joint_names)

        # Initialize the poses array
        poses = np.zeros((n_frames, n_joints, 3))

        # Reshape the data from long to wide format
        for frame in range(n_frames):
            frame_data = df[df['frame_id'] == frame]

            for joint_idx, joint_name in enumerate(self.joint_names):
                joint_data = frame_data[frame_data['joint_name'] == joint_name]
                if not joint_data.empty:
                    # Get the original coordinates
                    x = joint_data['x_3d'].iloc[0]
                    y = joint_data['y_3d'].iloc[0]
                    z = joint_data['z_3d'].iloc[0]

                    poses[frame, joint_idx] = [x, y, z ]

        return poses

    def convert_to_bvh(self, csv_path, output_path):
        """Convert 3d position CSV file to rotation angles BVH format"""
        # Convert CSV to numpy array
        print("🔄 Converting CSV to numpy array...")
        poses_3d = self.convert_csv_to_numpy(csv_path) # array num_frames, 17*3

        # Initialize H36M skeleton
        h36m_skel = h36m_skeleton.H36mSkeleton()

        # Convert to BVH
        channels, header = h36m_skel.poses2bvh(poses_3d, output_file=output_path)

        return channels, header

    def convert_to_csv(self, csv_path, output_path):
        """Convert 3d postion CSV file to appropriate rep csv file"""
        # Convert input CSV to numpy array
        poses_3d = self.convert_csv_to_numpy(csv_path)

        # Initialize H36M skeleton
        h36m_skel = h36m_skeleton.H36mSkeleton()

        # channels= h36m_skel.poses2expmap_csv(poses_3d, output_file=output_path)
        channels = h36m_skel.poses2quat_csv(poses_3d, output_file=output_path)

        columns = []
        for joint_idx, joint_name in enumerate(self.joint_names):
            if joint_name == "Hip":
                columns.append(joint_name + "_xposition")
                columns.append(joint_name + "_yposition")
                columns.append(joint_name + "_zposition")

            columns.append(joint_name + "_w") # for quaternian only
            columns.append(joint_name+"_x")
            columns.append(joint_name+"_y")
            columns.append(joint_name +"_z")


        # Create the DataFrame
        df = pd.DataFrame(channels, columns=columns)
        df.to_csv(output_path, index=False)
        print(f"file saved to {output_path}")

        return df


    def plot_pos_rep(self, csv_path, output_path):
        poses_3d = self.convert_csv_to_numpy(csv_path)
        # poses = np.zeros((n_frames, n_joints, 3))
        # j=1 0-2 j-1:j+2
        # j=2 3-5
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        joint_to_show = ["LHip", "LKnee","LWrist","LElbow"]
        plot_count = 0
        fig, axes = plt.subplots(len(joint_to_show), 1, figsize=(16, 10))
        coordinates = ["xposition", "yposition", "zposition"]
        for joint_idx, joint_name in enumerate(self.joint_names):
            if joint_name in joint_to_show:

                data = poses_3d[:, joint_idx, :].reshape(-1, 3)
                print(data.shape)
                print(joint_name)

                ax = axes[plot_count]
                ax.set_title(f'{joint_name} joint position',
                                 fontsize=16, fontweight='bold')

                for idx,value in enumerate(coordinates):
                    color = colors[idx % len(colors)]
                    ax.plot(range(data.shape[0]), data[:,idx],
                            color=color,
                            label=value,
                            linewidth=1.5,
                            alpha=0.7)

                    ax.set_xlabel('Frames', fontsize=12)
                    ax.set_ylabel('position', fontsize=12)
                    ax.legend(fontsize=10, loc='upper right')
                    ax.grid(True, alpha=0.3)

                plot_count += 1


        plt.tight_layout()
        figures_dir = os.path.join("./../../results", 'position_rep_visualization')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, output_path),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"file saved to {os.path.join(figures_dir, output_path)}")

    def convert_position_to_csv(self, csv_path, output_path):
        """Convert 3d postion CSV file to appropriate rep csv file"""
        # Convert input CSV to numpy array
        poses_3d = self.convert_csv_to_numpy(csv_path)


        columns = []
        data = []
        for joint_idx, joint_name in enumerate(self.joint_names):

            columns.append(joint_name + "_x")
            columns.append(joint_name + "_y")
            columns.append(joint_name + "_zs")
            # data.extend(poses_3d[:, joint_idx, :].reshape(-1, 3)
            #     poses_3d[:, :, :].reshape(-1, 3)

        for frame, pose in enumerate(poses_3d):
            data_frame = []
            for joint_idx, joint_name in enumerate(self.joint_names):
                data_frame.extend(pose[joint_idx,:])
            data.append(data_frame)
        # Create the DataFrame
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_path, index=False)
        print(f"file saved to {output_path}")

        return df
