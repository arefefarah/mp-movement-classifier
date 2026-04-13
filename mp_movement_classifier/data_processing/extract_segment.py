
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json
import pandas as pd

from mp_movement_classifier.utils.utils import read_bvh_files, process_bvh_data,process_motion_data

data_dir = "../../data/pymotion_position_csv_files"

path = Path("../../data/bvh_files")
csv_files = list(path.glob("*.csv"))
filename = "subject_11_motion_14.csv"
csv_file_path = Path(data_dir) / filename

motion_df = pd.read_csv(csv_file_path)
Segments_index= './data/segments_index.json'
with open(Segments_index, 'r') as f:
    data = json.load(f)
    boundaries = data[csv_file_path.stem]
print(boundaries)

segments = []
for boundary in boundaries:
    seg_df = motion_df.iloc[boundary[0]:boundary[1], :]
    segments.append(seg_df.to_numpy())

frame_rate = 30
frame_time = 1 / frame_rate

boundary_frames = [boundaries[0][0]] + [b[1] for b in boundaries]

 # Create time vector
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
    for boundary in boundaries:  # Exclude first and last
        # ax.axvline(x=time_vector[boundary], color='r', linestyle='--', alpha=0.7)
        ax.axvline(x=time_vector[boundary[0]], color='r', linestyle='--', alpha=0.7)

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
save_dir = "../../results/segmentation_analysis"
figures_dir = os.path.join(save_dir, "boundaries_plot")
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, f"{csv_file_path.stem}.png"),
            dpi=300, bbox_inches='tight')
plt.close()

