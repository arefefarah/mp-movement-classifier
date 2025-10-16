import matplotlib.pyplot as plt
import numpy
import numpy as np

import os
from matplotlib.patches import Patch

###### Plotting code ######

def set_figures_directory(new_figures_dir: str) -> None:
    """
    Set the global figures directory for all plotting functions.
    """
    global figures_dir
    figures_dir = new_figures_dir
    os.makedirs(figures_dir, exist_ok=True)


def get_figures_directory() -> str:
    """
    Get the current figures directory.
    """
    return figures_dir

def plot_eigenvalues(prec_mat, numw, title="", save=False):
    cv_mat = numpy.linalg.inv(prec_mat)
    ev = numpy.linalg.eigvals(cv_mat[:numw, :numw])
    ev.sort()
    ev = ev[::-1]

    plt.clf()
    plt.plot(ev, linewidth=2)
    plt.title("EVs weights, " + title)
    plt.savefig(os.path.join(figures_dir, f"ev_weights_{numw}_{title}.png"))
    plt.close()  # Use close instead of show to prevent display and save resources

    ev = numpy.linalg.eigvals(cv_mat[numw:, numw:])
    ev.sort()
    ev = ev[::-1]
    plt.clf()
    plt.plot(ev, linewidth=2)
    plt.title("EVs MPs, " + title)

    if save:
        plt.savefig(os.path.join(figures_dir, f"ev_MPs_{numw}_{title}.png"))
    else:
        plt.show()
    plt.close()


def plot_mp(MPs, title="", save=False):
    MPs = MPs.detach().numpy()
    plt.clf()
    for i in range(MPs.shape[0]):
        plt.plot(MPs[i], linewidth=2, label=f"MP {i}")

    plt.suptitle("MPs: " + title)
    plt.xlabel("t")
    plt.ylabel("joint angles [rad]")
    plt.subplots_adjust(left=0.2)

    if save:
        # Create a safe filename by removing problematic characters
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        plt.savefig(os.path.join(figures_dir, f"MPs_{safe_title}.png"))
    else:
        plt.show()
    plt.close()


def plot_learn_curve(epochs, lc, vc, title="", save=False):
    plt.clf()
    plt.plot(epochs, lc, linewidth=2)
    plt.xlabel("epochs")
    plt.ylabel("log(p(X,W,MP))")
    plt.title("Learning curve: " + title)
    plt.subplots_adjust(left=0.2)
    if save:
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        plt.savefig(os.path.join(figures_dir, f"learning_curve_{safe_title}.png"))
    else:
        plt.show()

    plt.close()

    plt.clf()
    plt.plot(epochs, vc, linewidth=2)
    plt.xlabel("epochs")
    plt.ylabel("VAF")
    plt.title("VAF: " + title)
    plt.subplots_adjust(left=0.2)

    if save:
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        plt.savefig(os.path.join(figures_dir, f"VAF_curve_{safe_title}.png"))
    else:
        plt.show()
    plt.close()


def plot_reconstructions(orig, recon, title="", save=False):
    plt.clf()

    for i in range(6):  # I add 6 to i to make sure it plots Rhip and Rknee
        plt.subplot(2, 3, i + 1)
        if i == 0:
            plt.plot(orig[i + 6], linewidth=2, linestyle="dotted", label="data")
            plt.plot(recon[i + 6], linewidth=1, label="model")
            plt.legend()
        else:
            plt.plot(orig[i + 6], linewidth=2, linestyle="dotted")
            plt.plot(recon[i + 6], linewidth=1)

    plt.tight_layout()
    plt.suptitle("Reconstructions: " + title, y=1.02)
    if save:
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        plt.savefig(os.path.join(figures_dir, f"recon_{safe_title}.png"), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_kernel(K, save=False):
    kvals = K[len(K) // 2]
    idx = numpy.arange(len(K)) - len(K) // 2
    plt.clf()
    plt.plot(idx, kvals, linewidth=2)
    plt.xlabel("$\Delta t$")
    plt.ylabel("covariance")
    plt.title("Kernel function")
    plt.savefig(os.path.join(figures_dir, "kernel.png"))
    plt.close()

    plt.clf()
    plt.title("Kernel matrix")
    plt.imshow(K)
    plt.xlabel("$t$")
    plt.ylabel("$t^\prime$")
    if save:
        plt.savefig(os.path.join(figures_dir, "kernel_matrix.png"))
    else:
        plt.show()

    plt.close()


def plot_model_comparison(model_evidences, VAFs, ground_truth_num_MPs, title="", save=False):
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.bar(range(1, 10), model_evidences)
    plt.ylabel("LAP")

    plt.subplot(1, 2, 2)
    plt.bar(range(1, 10), VAFs)
    plt.ylabel("VAF")

    plt.suptitle(f"Model comparison (Ground truth: {ground_truth_num_MPs} MPs)")
    if save:
        plt.savefig(os.path.join(figures_dir, "model_comparison.png"))
    else:
        plt.show()

    plt.close()


def plot_weights_for_signal(model, signal_idx, title=None, save=False):
    # Get number of segments and MPs
    num_segments = len(model.weights)
    num_MPs = model.num_MPs

    # Create figure
    plt.figure(figsize=(8, 4))

    # For each MP, collect weights from all segments for the chosen signal
    for mp_idx in range(num_MPs):
        # Extract weights for this MP and signal across all segments
        weights = [model.weights[seg_idx][signal_idx, mp_idx].item() for seg_idx in range(num_segments)]

        # Create x positions for this MP (add small jitter to separate points)
        x_positions = np.random.normal(mp_idx + 1, 0.05, len(weights))

        # Plot points for this MP
        plt.scatter(x_positions, weights, label=f'MP {mp_idx + 1}', alpha=0.7)

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Set plot labels and title
    plt.xlabel('Movement Primitive (MP)')
    plt.ylabel(f'Weight for Signal {signal_idx + 1}')
    if title:
        plt.title(title)
    else:
        plt.title(f'Weights for Signal(joint) {signal_idx + 1} across all segments')

    # Set x-ticks to MP numbers
    plt.xticks(range(1, num_MPs + 1))

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(figures_dir, f'weights_signal_{signal_idx + 1}.png'))
    else:
        plt.show()

    plt.close()


def plot_weights_by_joint(model, motion_label, save=False):
    """
    Plot weight distribution for all signals grouped by joint, with one subplot per MP.

    Parameters:
    -----------
    model : MP_model
        The trained temporal movement primitive model
    motion_label : str
        Label for the motion type, used in saving the figure
    """
    # Get dimensions
    num_segments = len(model.weights)
    num_MPs = model.num_MPs
    num_signals = model.weights[0].shape[0]

    # Get signal names and mapping
    # signal_names, signal_mapping = identify_signals(num_signals)
    signal_names, signal_mapping = identify_reduced_signals(num_signals)

    # Create figure with subplots
    fig, axes = plt.subplots(num_MPs, 1, figsize=(18, 4 * num_MPs), sharex=True)
    if num_MPs == 1:
        axes = [axes]  # Make it iterable for a single subplot

    # Define colors for different signal types
    signal_colors = {
        # 'Xposition': 'tab:red',
        # 'Yposition': 'tab:green',
        # 'Zposition': 'tab:blue',
        'Zrotation': 'tab:purple',
        'Xrotation': 'tab:orange',
        'Yrotation': 'tab:brown'
    }

    # For each MP
    for mp_idx in range(num_MPs):
        ax = axes[mp_idx]

        # For each signal, collect weights from all segments
        for signal_idx in range(num_signals):
            signal_name = signal_mapping[signal_idx]
            signal_type = signal_name.split('_')[1]

            # Extract weights for this MP and signal across all segments
            weights = [model.weights[seg_idx][signal_idx, mp_idx].item() for seg_idx in range(num_segments)]

            # Create x positions for this signal
            x_positions = np.random.normal(signal_idx + 1, 0.1, len(weights))

            # Plot points for this signal
            ax.scatter(x_positions, weights, alpha=0.6, s=30, color=signal_colors.get(signal_type, 'gray'))

        # Add a horizontal line at y=0 for reference
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Set plot labels
        ax.set_ylabel(f'Weights for MP {mp_idx + 1}')
        ax.set_title(f'Movement Primitive {mp_idx + 1} Weight Distribution')

        ax.grid(True, alpha=0.3)

    plt.xlabel('Joint Signal')

    # Create custom x-tick labels
    tick_positions = list(range(1, num_signals + 1))
    tick_labels = [f"{signal_idx + 1}: {name}" for signal_idx, name in signal_mapping.items()]

    # Use every few ticks to avoid overcrowding
    step = max(1, num_signals // 20)
    plt.xticks(tick_positions[::step], tick_labels[::step], rotation=45, ha='right', fontsize=8)

    # Add a legend for signal types
    legend_elements = [Patch(facecolor=color, label=signal_type)
                       for signal_type, color in signal_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', title='Signal Types')

    fig.suptitle(f'Weight Distribution by Joint in {motion_label}', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust to make room for the suptitle and rotated labels
    if save:
        safe_label = "".join(c for c in motion_label if c.isalnum() or c in (' ', '-', '_')).rstrip()
        plt.savefig(os.path.join(figures_dir, f'weights_{safe_label}.png'))
    else:
        plt.show()
    plt.close()


# Keep the existing identify_signals and other functions as they were
def identify_signals(num_signals=54):
    """
    Create a mapping between signal indices and joint names with rotation/position axes
    based on Human3.6M keypoint structure.

    Parameters:
    -----------
    num_signals : int
        Total number of signals (default=54)

    Returns:
    --------
    signal_names : list
        List of signal names in format "JointName_Axis"
    signal_mapping : dict
        Dictionary mapping signal indices to signal names
    """
    # Human3.6M keypoint names
    H36M_KEYPOINT_NAMES = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Thorax', 'Neck', 'Head',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
    ]

    # Create signal names
    signal_names = []
    signal_mapping = {}

    signal_idx = 0

    # Hip has 6 signals: X/Y/Z position and Z/X/Y rotation
    hip_signals = ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']
    for signal in hip_signals:
        signal_name = f"Hip_{signal}"
        signal_names.append(signal_name)
        signal_mapping[signal_idx] = signal_name
        signal_idx += 1

    # All other joints have 3 rotation signals: Z/X/Y rotation
    rotation_signals = ['Zrotation', 'Xrotation', 'Yrotation']

    for joint in H36M_KEYPOINT_NAMES[1:]:  # Skip Hip as it's already processed

        for signal in rotation_signals:
            signal_name = f"{joint}_{signal}"
            signal_names.append(signal_name)
            signal_mapping[signal_idx] = signal_name
            signal_idx += 1

    # Verify we have the expected number of signals
    expected_signals = 6 + (len(H36M_KEYPOINT_NAMES) - 1) * 3  # 6 for Hip + 3 for each other joint

    if expected_signals != num_signals:
        print(f"Warning: Expected {expected_signals} signals based on H36M structure.")
        print(f"Actual number of signals: {num_signals}")

        # Fill any remaining indices if needed
        for i in range(signal_idx, num_signals):
            signal_name = f"Unknown_{i}"
            signal_names.append(signal_name)
            signal_mapping[i] = signal_name

    return signal_names, signal_mapping


def identify_reduced_signals(num_signals=39):
    """
    Create a mapping between new signal indices and joint names with rotation/position axes
    """
    # updated keypoint based on 39 signals
    UPDATED_KEYPOINT_NAMES = [
        'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Head', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist']

    # Create signal names
    signal_names = []
    signal_mapping = {}
    signal_idx = 0

    rotation_signals = ['Zrotation', 'Xrotation', 'Yrotation']

    for joint in UPDATED_KEYPOINT_NAMES[0:]:

        for signal in rotation_signals:
            signal_name = f"{joint}_{signal}"
            signal_names.append(signal_name)
            signal_mapping[signal_idx] = signal_name
            signal_idx += 1

    return signal_names, signal_mapping


# def get_joint_channel_mapping():
#     """
#     Define channel mapping for BVH structure based on hierarchy
#     Returns dict with joint names and their channel indices
#     """
#     joint_mapping = {
#         # Hip: 6 channels (3 position + 3 rotation)
#         'Hip': {'pos': [0, 1, 2], 'rot': [3, 4, 5]},
#
#         # Right side joints (3 rotation channels each)
#         'RightHip': {'rot': [6, 7, 8]},
#         'RightKnee': {'rot': [9, 10, 11]},
#         'RightAnkle': {'rot': [12, 13, 14]},
#
#         # Left side joints (3 rotation channels each)
#         'LeftHip': {'rot': [15, 16, 17]},
#         'LeftKnee': {'rot': [18, 19, 20]},
#         'LeftAnkle': {'rot': [21, 22, 23]},
#
#         # Spine chain (3 rotation channels each)
#         'Spine': {'rot': [24, 25, 26]},
#         'Thorax': {'rot': [27, 28, 29]},
#         'Neck': {'rot': [30, 31, 32]},
#
#         # Left arm (3 rotation channels each)
#         'LeftShoulder': {'rot': [33, 34, 35]},
#         'LeftElbow': {'rot': [36, 37, 38]},
#         'LeftWrist': {'rot': [39, 40, 41]},
#
#         # Right arm (3 rotation channels each)
#         'RightShoulder': {'rot': [42, 43, 44]},
#         'RightElbow': {'rot': [45, 46, 47]},
#         'RightWrist': {'rot': [48, 49, 50]}
#     }
#
#     return joint_mapping
#
#
# def compute_movement_speed(data, joint_indices):
#     """
#     Compute movement speed for joints (angular velocity for rotation data)
#
#     Args:
#         data: motion data array [frames, channels]
#         joint_indices: list of channel indices for the joint
#
#     Returns:
#         speed: combined speed measure for the joint
#     """
#     if not joint_indices:
#         return np.zeros(data.shape[0])
#
#     # Calculate angular velocity (first derivative) for each rotation channel
#     speeds = []
#     for idx in joint_indices:
#         if idx < data.shape[1]:
#             # Angular velocity approximation
#             angular_velocity = np.gradient(data[:, idx])
#             speeds.append(np.abs(angular_velocity))
#
#     if speeds:
#         # Sum speeds across all channels for this joint
#         return np.sum(speeds, axis=0)
#     else:
#         return np.zeros(data.shape[0])
#

def process_bvh_data_with_info(bvh_data, num_points=50):
    """
    Same as process_bvh_data but returns additional segmentation information

    Returns:
        processed_segments: list of segmented motion data
        segmentation_info: list of dicts with segmentation details for each file
    """
    processed_segments = []
    segmentation_info = []

    for i, mocap in enumerate(bvh_data):
        # Convert frames to numpy array
        frames = np.array(mocap.frames, dtype=np.float64)

        # Get frame time
        try:
            frame_time = mocap.frame_time
        except:
            frame_time = 0.03333  # Default 30 FPS

        # Apply Butterworth filter
        filtered_frames = filter_motion_data(frames)

        # Apply temporal segmentation
        segments, boundaries = segment_motion_sequence(filtered_frames, frame_time)

        # Store segmentation info
        file_info = {
            'file_index': i,
            'original_frames': frames.shape[0],
            'num_segments': len(segments),
            'boundaries': boundaries,
            'segment_lengths': [seg.shape[0] for seg in segments]
        }
        segmentation_info.append(file_info)

        # Add segments to processed list
        for segment in segments:
            if segment.shape[0] > 10:  # Only keep segments with sufficient length
                processed_segments.append(segment.T)  # Transpose to [signals, time]

    if not processed_segments:
        raise ValueError("No segments could be processed")

    return processed_segments, segmentation_info


def remove_signals(processed_data):
    """
    Remove signals with names containing 'Hip', 'Neck', 'Spine', or 'Thorax'

    Returns:
    --------
    filtered_data : list
        List of numpy arrays with filtered signals
    excluded_indices : list
        List of indices that were excluded
    """
    # Get total number of signals
    num_signals = processed_data[0].shape[0]

    signal_names, signal_mapping = identify_signals(num_signals)

    exclude_joints = ['Hip_', 'Neck_', 'Spine_', 'Thorax_']

    excluded_indices = []
    for idx, name in signal_mapping.items():
        if any(name.startswith(joint) for joint in exclude_joints):
            excluded_indices.append(idx)

    excluded_indices.sort()

    keep_indices = [i for i in range(num_signals) if i not in excluded_indices]

    filtered_data = []
    for segment in processed_data:
        filtered_segment = segment[keep_indices, :]
        filtered_data.append(filtered_segment)

    # Calculate and print summary
    kept_signals = num_signals - len(excluded_indices)
    print(f"Removed {len(excluded_indices)} signals. Keeping {kept_signals} signals.")

    return filtered_data, excluded_indices