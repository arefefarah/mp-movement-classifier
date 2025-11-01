import pandas as pd
import numpy as np
from pathlib import Path
import json
from bvh_skeleton import h36m_skeleton


class H36MConverter:
    def __init__(self):
        # Your skeleton structure
        self.joint_names = [
            'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
            'Spine', 'Thorax', 'Neck', 'Head',
            'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
        ]

        # Mapping from your joint names to H36M original skeleton joint names
        self.joint_mapping = {
            'Hip': 'Hip',
            'RHip': 'RightUpLeg',
            'RKnee': 'RightLeg',
            'RAnkle': 'RightFoot',
            'LHip': 'LeftUpLeg',
            'LKnee': 'LeftLeg',
            'LAnkle': 'LeftFoot',
            'Spine': 'Spine',
            'Thorax': 'Spine1',
            'Neck': 'Neck',
            'Head': 'Head',
            'LShoulder': 'LeftShoulder',
            'LElbow': 'LeftArm',
            'LWrist': 'LeftHand',
            'RShoulder': 'RightShoulder',
            'RElbow': 'RightArm',
            'RWrist': 'RightHand'
        }

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
        """Convert CSV file to BVH format"""
        # Convert CSV to numpy array
        poses_3d = self.convert_csv_to_numpy(csv_path)

        # Initialize H36M skeleton
        h36m_skel = h36m_skeleton.H36mSkeleton()

        # Convert to BVH
        channels, header = h36m_skel.poses2bvh(poses_3d, output_file=output_path)

        return channels, header

