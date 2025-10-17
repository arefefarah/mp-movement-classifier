import glob
from scipy import signal
import cv2
import pandas as pd
import json
import torch
import os
import numpy as np
import re
from bvh import Bvh
from scipy.signal import find_peaks

from TMP_model import MP_model
from mp_movement_classifier.utils import config


H36M_KEYPOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

# Define connections between joints for visualization (based on Human3.6M skeleton)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 4),       # Hip to RHip, Hip to LHip
    (1, 2), (2, 3),       # Right leg
    (4, 5), (5, 6),       # Left leg
    (0, 7), (7, 8),       # Spine to thorax
    (8, 9), (9, 10),      # Thorax to head
    (8, 11), (11, 12), (12, 13),  # Left arm
    (8, 14), (14, 15), (15, 16)   # Right arm
]



# Global dictionary to persist motion mappings across function calls
MOTION_MAPPING = {}
MOTION_ID_COUNTER = 0
MAPPING_FILE = "../../data/motion_mapping.json"

def load_existing_mapping():
    """Load existing motion mapping from file if it exists"""
    global MOTION_MAPPING, MOTION_ID_COUNTER
    try:
        if os.path.exists(MAPPING_FILE):
            with open(MAPPING_FILE, 'r') as f:
                data = json.load(f)
                MOTION_MAPPING = data["mapping"]
                MOTION_ID_COUNTER = data["counter"]
                print(f"Loaded existing motion mapping with {len(MOTION_MAPPING)} entries.")
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        MOTION_MAPPING = {}
        MOTION_ID_COUNTER = 0

def get_joint_channel_mapping():
    """
    Define channel mapping for BVH structure based on hierarchy
    Returns dict with joint names and their channel indices
    """
    joint_mapping = {
        # Hip: 6 channels (3 position + 3 rotation)
        'Hip': {'pos': [0, 1, 2], 'rot': [3, 4, 5]},

        # Right side joints (3 rotation channels each)
        'RightHip': {'rot': [6, 7, 8]},
        'RightKnee': {'rot': [9, 10, 11]},
        'RightAnkle': {'rot': [12, 13, 14]},

        # Left side joints (3 rotation channels each)
        'LeftHip': {'rot': [15, 16, 17]},
        'LeftKnee': {'rot': [18, 19, 20]},
        'LeftAnkle': {'rot': [21, 22, 23]},

        # Spine chain (3 rotation channels each)
        'Spine': {'rot': [24, 25, 26]},
        'Thorax': {'rot': [27, 28, 29]},
        'Neck': {'rot': [30, 31, 32]},

        # Left arm (3 rotation channels each)
        'LeftShoulder': {'rot': [33, 34, 35]},
        'LeftElbow': {'rot': [36, 37, 38]},
        'LeftWrist': {'rot': [39, 40, 41]},

        # Right arm (3 rotation channels each)
        'RightShoulder': {'rot': [42, 43, 44]},
        'RightElbow': {'rot': [45, 46, 47]},
        'RightWrist': {'rot': [48, 49, 50]}
    }

    return joint_mapping


def compute_movement_speed(data, joint_indices):
    """
    Compute movement speed for joints (angular velocity for rotation data)

    Args:
        data: motion data array [frames, channels]
        joint_indices: list of channel indices for the joint

    Returns:
        speed: combined speed measure for the joint
    """
    if not joint_indices:
        return np.zeros(data.shape[0])

    # Calculate angular velocity (first derivative) for each rotation channel
    speeds = []
    for idx in joint_indices:
        if idx < data.shape[1]:
            # Angular velocity approximation
            angular_velocity = np.gradient(data[:, idx])
            speeds.append(np.abs(angular_velocity))

    if speeds:
        # Sum speeds across all channels for this joint
        return np.sum(speeds, axis=0)
    else:
        return np.zeros(data.shape[0])



def read_bvh_files(folder_path):
    bvh_data = []
    motion_ids = []
    bvh_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.bvh')]
    # Regular expression to extract the motion ID
    # This pattern looks for "motion_" followed by one or more digits
    pattern = re.compile(r'motion_(\d+)')

    for bvh_file in bvh_files:
        file_path = os.path.join(folder_path, bvh_file)
        try:
            match = pattern.search(bvh_file)
            if match:
                motion_id = int(match.group(1))  # Convert to integer
                # Read the BVH file
                with open(file_path, 'r', encoding='utf-8') as f:
                    mocap = Bvh(f.read())
                    bvh_data.append(mocap)
                    motion_ids.append(motion_id)
            else:
                print(f"Could not extract motion ID from filename: {bvh_file}")
        except Exception as e:
            print(f"Error reading file {bvh_file}: {str(e)}")
    return bvh_data, motion_ids

def filter_motion_data(data, cutoff_freq=6, sampling_rate=30):
    """Apply 4th order Butterworth filter as specified in paper"""
    nyquist_freq = 0.5 * sampling_rate
    order = 4
    # cutoff_freq = nyquist_freq * 0.8
    b, a = signal.butter(order, cutoff_freq / nyquist_freq)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def process_bvh_data(bvh_data, motion_ids, cutoff_freq=3):
    """
    Apply 4th order Butterworth filter and segmentation as specified in Leh et al. (2023)

    Args:
        bvh_data: list of BVH mocap objects
        motion_ids: list of motion IDs corresponding to each BVH sequence

    Returns:
        processed_segments: list of segmented motion data in [signals, time] format
        segment_motion_ids: list of motion IDs corresponding to each segment
    """
    processed_segments = []
    segment_motion_ids = []

    for mocap, motion_id in zip(bvh_data, motion_ids):
        # Convert frames to numpy array
        frames = np.array(mocap.frames, dtype=np.float64)

        # Get frame time
        try:
            frame_time = mocap.frame_time
        except:
            frame_time = 0.03333  # Default 30 FPS

        # Apply Butterworth filter (as specified in paper)
        filtered_frames = filter_motion_data(frames,cutoff_freq=cutoff_freq)

        # Apply temporal segmentation based on limb endpoint movement
        segments, boundaries = segment_motion_sequence(filtered_frames, frame_time)

        # Add each segment to processed segments in [signals, time] format
        for segment in segments:
            if segment.shape[0] > 10:  # Only keep segments with sufficient length
                processed_segments.append(segment.T)  # Transpose to [signals, time]
                segment_motion_ids.append(motion_id)

    if not processed_segments:
        raise ValueError("No segments could be processed")

    return processed_segments, segment_motion_ids

def segment_motion_sequence(data, frame_time=0.03333, min_segment_length=160e-3):
    """
    Segment motion based on Leh et al. (2023) approach:
    - Use minima of summed speed of limb endpoints (wrists and ankles)
    - Minimum distance of 160ms between boundaries

    Args:
        data: filtered motion data [frames, channels]
        frame_time: time between frames in seconds
        min_segment_length: minimum time between boundaries in seconds

    Returns:
        segments: list of data segments
        boundaries: frame indices of segment boundaries
    """
    joint_mapping = get_joint_channel_mapping()

    # Define limb endpoints for segmentation (following Leh et al.)
    limb_endpoints = ['LeftWrist', 'RightWrist', 'LeftAnkle', 'RightAnkle']

    # Compute combined speed of limb endpoints
    total_speed = np.zeros(data.shape[0])

    for joint_name in limb_endpoints:
        if joint_name in joint_mapping:
            joint_indices = joint_mapping[joint_name]['rot']
            joint_speed = compute_movement_speed(data, joint_indices)
            total_speed += joint_speed

    # Smooth the speed signal slightly to reduce noise in boundary detection
    if len(total_speed) > 5:
        total_speed = signal.medfilt(total_speed, kernel_size=3)

    # Find minima in the combined speed
    # Invert signal to find minima as peaks
    inverted_speed = -total_speed

    # Convert minimum segment length to frames
    min_frames = int(min_segment_length / frame_time)

    # Find peaks (which are minima in original signal)
    if len(inverted_speed) > min_frames:
        peaks, _ = find_peaks(inverted_speed, distance=min_frames, height=None)
    else:
        peaks = []

    # Always include start and end boundaries
    boundaries = [0]
    boundaries.extend(peaks)
    boundaries.append(data.shape[0] - 1)

    # Remove duplicates and sort
    boundaries = sorted(list(set(boundaries)))

    # Ensure minimum distance between boundaries
    filtered_boundaries = [boundaries[0]]
    for i in range(1, len(boundaries)):
        if boundaries[i] - filtered_boundaries[-1] >= min_frames:
            filtered_boundaries.append(boundaries[i])

    # Always include the last boundary if not already there
    if filtered_boundaries[-1] != boundaries[-1]:
        filtered_boundaries.append(boundaries[-1])

    # Create segments
    segments = []
    for i in range(len(filtered_boundaries) - 1):
        start_idx = filtered_boundaries[i]
        end_idx = filtered_boundaries[i + 1]
        segment = data[start_idx:end_idx, :]
        if segment.shape[0] > 0:  # Only add non-empty segments
            segments.append(segment)

    return segments, filtered_boundaries



def save_mapping():
    """Save the current motion mapping to file"""
    try:
        with open(MAPPING_FILE, 'w') as f:
            json.dump({
                "mapping": MOTION_MAPPING,
                "counter": MOTION_ID_COUNTER
            }, f, indent=2)
        print(f"Saved motion mapping with {len(MOTION_MAPPING)} entries.")
    except Exception as e:
        print(f"Error saving mapping file: {e}")

def create_motion_mapping(motions_list):
    """
    Create a mapping from motion sequence numbers to standardized motion IDs.
    If a motion is new, it gets a new ID. If it's already known, it keeps its ID.
    
    Parameters:
        motions_list: List of motion names from the .mat file
        
    Returns:
        Dictionary mapping sequence numbers to motion IDs
    """
    global MOTION_MAPPING, MOTION_ID_COUNTER
    
    # Load existing mapping if it's the first call
    if not MOTION_MAPPING:
        load_existing_mapping()
    
    # Create a mapping for this specific file
    sequence_to_id = {}
    print("Available motions in this file:")
    
    for i, motion in enumerate(motions_list):
        motion_name = str(motion).lower().strip()
        # print(f"  {i+1}: {motion_name}")
        
        # Check if this motion is already in our mapping
        if motion_name in MOTION_MAPPING:
            motion_id = MOTION_MAPPING[motion_name]
        else:
            # New motion, assign a new ID
            motion_id = MOTION_ID_COUNTER
            MOTION_MAPPING[motion_name] = motion_id
            MOTION_ID_COUNTER += 1
            print(f"  Added new motion: '{motion_name}' with ID {motion_id}")
        
        sequence_to_id[i+1] = motion_id
    
    # Save the updated mapping
    save_mapping()
   
    return sequence_to_id

def single_videos(main_video, main_rub):
    """
    Extract the single video motions using "flags" field from
    the corresponding rub file and save them in a folder named after the original video.
    Files are named using subject ID and a consistent motion ID.
    
    Parameters:
        main_video (str): The full name of the main video file which contains all motions
        main_rub (dict): Corresponding rub style struct (loaded from .mat file)
    """
    
    move_num = 1
    video_obj = cv2.VideoCapture(main_video)
    total_frames = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    
    filepath, filename = os.path.split(main_video)
    name, ext = os.path.splitext(filename)
    
    # Extract subject ID from the filename using regex
    subject_id_match = re.search(r'Subject_(\d+)', filename)
    if subject_id_match:
        subject_id = subject_id_match.group(1)
    else:
        # If no match, use the full filename
        subject_id = name
    
    # Create a directory with the same name as the original video
    output_dir = os.path.join(filepath, name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Based on the output, 'Subject_20_F' is a key in main_rub
    subject_key = list(filter(lambda k: k not in ['__header__', '__version__', '__globals__'], main_rub.keys()))[0]
    subject_struct = main_rub[subject_key]
    
    
    try:
        main_move = subject_struct.move
        
        if hasattr(main_move, 'flags30'):
            flags30 = main_move.flags30
            # print("Found flags30 directly in move")
        else:
            # The move might be a struct with numbered fields
            for i in range(1, 10):  # Try reasonable field names
                field_name = str(i)
                if hasattr(main_move, field_name):
                    sub_move = getattr(main_move, field_name)
                    if hasattr(sub_move, 'flags30'):
                        flags30 = sub_move.flags30
                        # print(f"Found flags30 in move.{field_name}")
                        break
            else:
                raise KeyError("Could not find flags30 field")
        
        # Try to get the motions_list from the same place as flags30
        if hasattr(main_move, 'motions_list'):
            motions_list = main_move.motions_list
            num_motions = len(motions_list)
            
            motion_mapping = create_motion_mapping(motions_list)
                
        else:
            # If there's no motions_list, just use the number of segments in flags30
            num_motions = len(flags30)
            
            # Create a simple mapping without motion names
            # We'll just use sequential IDs starting from the current counter
            sequence_to_id = {}
            for i in range(num_motions):
                sequence_to_id[i+1] = MOTION_ID_COUNTER + i
            
            # Update the counter for next time
            MOTION_ID_COUNTER += num_motions
            
            motion_mapping = sequence_to_id
        
    except (AttributeError, KeyError) as e:
        print(f"Error accessing move structure: {e}")
        raise
 
    
    # Reading the frames one by one and saving the videos
    counter = 1
    frame_idx = 0
    
    while frame_idx < total_frames:
        ret, frame = video_obj.read()
        if not ret:
            break
            
        frame_idx += 1  # Python uses 0-indexing but we need 1-indexing to match MATLAB
        
        # Check if current frame is within a motion segment
        if counter <= num_motions:
            start_frame = flags30[counter-1][0] if isinstance(flags30[counter-1], np.ndarray) else flags30[counter-1, 0]
            end_frame = flags30[counter-1][1] if isinstance(flags30[counter-1], np.ndarray) else flags30[counter-1, 1]
            
            if frame_idx >= start_frame and frame_idx <= end_frame:
                if frame_idx == start_frame:
                    # Get the motion ID from our mapping
                    motion_id = motion_mapping.get(counter, counter-1)
                    
                    # Format: subject_id_motion_id.avi
                    output_videofilename = os.path.join(output_dir, f"subject_{subject_id}_motion_{motion_id:02d}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fps = 30
                    frame_size = (frame.shape[1], frame.shape[0])
                    output_video = cv2.VideoWriter(output_videofilename, fourcc, fps, frame_size)
                    print(f"Creating video: {output_videofilename}")
                
                output_video.write(frame)
                
                if frame_idx == end_frame:
                    motion_id = motion_mapping.get(counter, counter-1)
                    counter += 1
                    output_video.release()
                    # print(f"Finished writing motion ID {motion_id:02d} for subject {subject_id}")
                    if counter > num_motions:
                        break
    
    video_obj.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. All segments saved to {output_dir}")


def create_csv_from_json(source_folder, destination_folder, subjects, equivalent_motions, id_to_motion):
    """
    Function to create CSV files from JSON motion files
    
    Args:
        source_folder: Directory containing subject folders with JSON files
        destination_folder: Directory to save individual CSV files
        subjects: List of subject IDs to process
        equivalent_motions: Dictionary mapping motion IDs to canonical IDs
        id_to_motion: Dictionary mapping motion IDs to motion names
    
    Returns:
        dict: Dictionary mapping subject IDs to their common motions
        list: List of motion IDs common to all subjects
    """
    os.makedirs(destination_folder, exist_ok=True)
    
    # Find subjects and their available motions
    subject_motions = {}

    for subject_id in subjects:
        subject_folder = os.path.join(source_folder, f"pred_Subject_{subject_id}")
        if os.path.exists(subject_folder):
            json_files = glob.glob(os.path.join(subject_folder, "*.json"))
            motion_ids = set()  # Using a set to automatically handle duplicates
            
            for json_file in json_files:
                filename = os.path.basename(json_file)
                # Extract motion ID from filename
                match = re.search(f"subject_{subject_id}_motion_(\\d+)", filename)
                if match:
                    motion_id = match.group(1)
                    # Convert to canonical ID if it's an equivalent motion
                    canonical_id = equivalent_motions.get(str(motion_id), str(motion_id))
                    
                    motion_ids.add(int(canonical_id))
            
            subject_motions[subject_id] = list(motion_ids)

    # Find common motions across all subjects
    all_motion_sets = [set(motions) for motions in subject_motions.values()]
    if all_motion_sets:
        common_motions = set.intersection(*all_motion_sets)
        common_motions = sorted([int(m) for m in common_motions])
        print(f"Common motions across all subjects: {common_motions}")
        print(f"Motion names: {[id_to_motion.get(str(m), f'Unknown-{m}') for m in common_motions]}")
    else:
        common_motions = []
        print("No subjects with motions found.")

    # Process each subject
    for subject_id in subjects:
        subject_folder = os.path.join(source_folder, f"pred_Subject_{subject_id}")
        
        if not os.path.exists(subject_folder):
            print(f"Subject folder not found: {subject_folder}")
            continue
        
        # Process each motion for this subject
        for canonical_motion_id in common_motions:
            canonical_motion_id_str = str(canonical_motion_id)
            
            # Find all possible equivalent motion IDs for this canonical ID
            possible_motion_ids = [motion_id for motion_id, canon_id in equivalent_motions.items() 
                                  if canon_id == canonical_motion_id_str]
            # Also include the canonical ID itself
            if canonical_motion_id_str not in possible_motion_ids:
                possible_motion_ids.append(canonical_motion_id_str)
            
            # Try all possible motion IDs
            found = False
            for motion_id in possible_motion_ids:
                # Check both padded and unpadded versions in file search, but always use padded for output
                padded_motion_id_2 = motion_id.zfill(2)
                padded_motion_id_1 = motion_id  # Original format
                
                # Try both padding formats for searching
                for padded_id in [padded_motion_id_2, padded_motion_id_1]:
                    motion_pattern = f"subject_{subject_id}_motion_{padded_id}.json"
                    motion_files = glob.glob(os.path.join(subject_folder, motion_pattern))
                    
                    if motion_files:
                        json_file_path = motion_files[0]
                        
                        # ALWAYS use zero-padded format (2 digits) for output files
                        padded_canonical_id = str(canonical_motion_id).zfill(2)
                        output_path = os.path.join(destination_folder, f"subject_{subject_id}_motion_{padded_canonical_id}.csv")
                        
                        # Process the JSON file and create CSV
                        with open(json_file_path, 'r') as f:
                            predictions = json.load(f)
                        
                        # Create empty lists to store the data
                        frame_ids = []
                        joint_names = []
                        x_3d = []
                        y_3d = []
                        z_3d = []
                        confidence = []
                        
                        # Process each frame
                        for frame_data in predictions:
                            frame_id = frame_data["frame_id"]
                            
                            # Check if there are instances in this frame
                            if 'instances' in frame_data and len(frame_data["instances"]) > 0:
                                # Get the first person (or you can loop through all if needed)
                                person_data = frame_data["instances"][0]
                                
                                # Extract 3D keypoints
                                keypoints_3d = person_data['keypoints']
                                scores = person_data['keypoint_scores']
                                
                                # Process each keypoint
                                for idx, (point, score) in enumerate(zip(keypoints_3d, scores)):
                                    if idx < len(H36M_KEYPOINT_NAMES):
                                        joint_name = H36M_KEYPOINT_NAMES[idx]
                                        
                                        frame_ids.append(frame_id)
                                        joint_names.append(joint_name)
                                        x_3d.append(point[0])
                                        y_3d.append(point[1])
                                        z_3d.append(point[2])
                                        confidence.append(score)
                        
                        # Get canonical motion name
                        motion_name = id_to_motion.get(canonical_motion_id_str, f'Unknown-{canonical_motion_id}')
                        
                        # Create a DataFrame with the 3D keypoint data
                        df_3d = pd.DataFrame({
                            'frame_id': frame_ids,
                            'joint_name': joint_names,
                            'x_3d': x_3d,
                            'y_3d': y_3d,
                            'z_3d': z_3d,
                            'confidence': confidence,
                            'motion_id': canonical_motion_id,
                            'original_motion_id': int(motion_id),
                            'motion_name': motion_name,
                            'subject_id': subject_id
                        })
                        
                        # Save the DataFrame to CSV
                        df_3d.to_csv(output_path, index=False)
                        
                        found = True
                        break
                
                if found:
                    break
    
    return subject_motions, common_motions

def merge_subject_csv_files(destination_folder, merged_folder, subjects, common_motions):
    """
    Function to merge CSV files for each subject
    
    Args:
        destination_folder: Directory containing individual CSV files
        merged_folder: Directory to save merged CSV files
        subjects: List of subject IDs to process
        common_motions: List of motion IDs common to all subjects
    """
    os.makedirs(merged_folder, exist_ok=True)
    
    for subject_id in subjects:
        # Dictionary to store dataframes for common motions
        subject_dfs = {}
        
        # Process each motion for this subject
        for canonical_motion_id in common_motions:
            # Try different padding formats for motion IDs in filenames
            found = False
            
            # Try with 2-digit padding (00, 01, etc.)
            padded_id = str(canonical_motion_id).zfill(2)
            csv_pattern = f"subject_{subject_id}_motion_{padded_id}.csv"
            csv_files = glob.glob(os.path.join(destination_folder, csv_pattern))
            
            if not csv_files:
                # Try with 1-digit padding (0, 1, etc.)
                padded_id = str(canonical_motion_id).zfill(1)
                csv_pattern = f"subject_{subject_id}_motion_{padded_id}.csv"
                csv_files = glob.glob(os.path.join(destination_folder, csv_pattern))
            
            if csv_files:
                # Load CSV file
                csv_file_path = csv_files[0]
                df = pd.read_csv(csv_file_path)
                subject_dfs[canonical_motion_id] = df
        
        # Merge all dataframes for this subject
        if subject_dfs:
            # Sort by motion_id to ensure consistent order
            sorted_dfs = [subject_dfs[motion_id] for motion_id in sorted(subject_dfs.keys()) if motion_id in subject_dfs]
            
            if sorted_dfs:  # Check if we have any dataframes to merge
                merged_df = pd.concat(sorted_dfs, ignore_index=True)
                
                # Save merged dataframe
                merged_output_path = os.path.join(merged_folder, f"subject_{subject_id}_all_motions.csv")
                merged_df.to_csv(merged_output_path, index=False)
                print(f"Created merged file for subject {subject_id} with {len(sorted_dfs)} motions")
            else:
                print(f"No common motions found for subject {subject_id}")

#
# def segmentation(motion_id,subject_id):
#
#     motion_id = motion_id
#     padded_motion_id = str(motion_id).zfill(2)
#
#     subject_id = subject_id
#     csv_file = f"../../data/MMpose/df_files_3d/subject_{subject_id}_motion_{padded_motion_id}.csv"
#
#     with open('../../data/common_motion_mapping.json', 'r') as f:
#         motion_mapping = json.load(f)['mapping']
#         id_to_motion = {str(v): k for k, v in motion_mapping.items()}
#
#     motion_name = id_to_motion.get(str(motion_id), f"motion_{motion_id}") if id_to_motion else f"motion_{motion_id}"
#
#     destination_folder = f"../../data/MMpose/segmented_files/{motion_name}"
#
#     os.makedirs(destination_folder, exist_ok=True)
#
#     df_3d = pd.read_csv(csv_file)
#     frames = df_3d['frame_id'].unique()
#     fps = 120  # Given in description
#
#     # Step 1: Extract wrist and foot markers positions per frame
#     # We need to reshape the data to get positions for each joint at each frame
#
#     # Get positions for relevant joints (LWrist, RWrist, LAnkle, RAnkle)
#     joint_positions = {}
#     for frame in frames:
#         frame_data = df_3d[df_3d['frame_id'] == frame]
#
#         # Initialize positions for this frame
#         joint_positions[frame] = {}
#
#         # Extract positions for each joint we need
#         for joint in ['LWrist', 'RWrist', 'LAnkle', 'RAnkle']:
#             joint_data = frame_data[frame_data['joint_name'] == joint]
#             if len(joint_data) > 0:
#                 # Get x, y, z coordinates
#                 joint_positions[frame][joint] = np.array([
#                     joint_data['x_3d'].values[0],
#                     joint_data['y_3d'].values[0],
#                     joint_data['z_3d'].values[0]
#                 ])
#             else:
#                 # Use zeros if joint data is missing
#                 joint_positions[frame][joint] = np.zeros(3)
#
#     # Convert to numpy arrays for easier processing
#     sorted_frames = sorted(frames)
#     lwrist_pos = np.array([joint_positions[frame]['LWrist'] for frame in sorted_frames])
#     rwrist_pos = np.array([joint_positions[frame]['RWrist'] for frame in sorted_frames])
#     lankle_pos = np.array([joint_positions[frame]['LAnkle'] for frame in sorted_frames])
#     rankle_pos = np.array([joint_positions[frame]['RAnkle'] for frame in sorted_frames])
#
#     # Step 2: Calculate velocities (difference between consecutive frames)
#     def calculate_velocity(positions):
#         # Calculate difference between consecutive positions
#         velocities = np.diff(positions, axis=0)
#         # Pad with a zero at the beginning to maintain array length
#         velocities = np.vstack([np.zeros((1, 3)), velocities])
#         return velocities
#
#     lwrist_vel = calculate_velocity(lwrist_pos)
#     rwrist_vel = calculate_velocity(rwrist_pos)
#     lankle_vel = calculate_velocity(lankle_pos)
#     rankle_vel = calculate_velocity(rankle_pos)
#
#     # Step 3: Calculate speeds (magnitude of velocity)
#     def calculate_speed(velocity):
#         return np.sqrt(np.sum(velocity**2, axis=1))
#
#     lwrist_speed = calculate_speed(lwrist_vel)
#     rwrist_speed = calculate_speed(rwrist_vel)
#     lankle_speed = calculate_speed(lankle_vel)
#     rankle_speed = calculate_speed(rankle_vel)
#
#     # Step 4: Sum the speeds of all markers
#     summed_speed = lwrist_speed + rwrist_speed + lankle_speed + rankle_speed
#
#     # Step 5: Find local minima in the summed speed
#     # Minimum distance between boundaries: 160ms = 0.16s * 120fps = 19.2 frames ≈ 19 frames
#     min_distance = int(0.16 * fps)  # 160ms converted to frames at 120fps
#
#     # Find local minima (as peaks in negative speed)
#     negative_speed = -summed_speed
#     minima_indices, _ = find_peaks(negative_speed, distance=min_distance)
#
#     # Add the first and last frame as boundaries
#     segment_boundaries = np.concatenate([[0], minima_indices, [len(summed_speed)-1]])
#
#     # Step 6: Visualize the segmentation
#     plt.figure(figsize=(15, 6))
#     plt.plot(summed_speed, label='Summed Speed')
#     plt.plot(minima_indices, summed_speed[minima_indices], 'ro', label='Detected Minima')
#     plt.vlines(segment_boundaries, ymin=0, ymax=max(summed_speed), colors='g', linestyles='dashed', label='Segment Boundaries')
#     plt.legend()
#     plt.title(f'Movement Segmentation for {motion_name}')
#     plt.xlabel('Frame')
#     plt.ylabel('Summed Speed')
#     plt.savefig(os.path.join(destination_folder, f'subject_{subject_id}_segmentation.png'))
#     plt.close()
#
#     # Step 7: Create segments based on boundaries
#     segments = []
#     for i in range(len(segment_boundaries)-1):
#         start_idx = int(segment_boundaries[i])
#         end_idx = int(segment_boundaries[i+1])
#
#         # Get the actual frame IDs for this segment
#         start_frame = sorted_frames[start_idx]
#         end_frame = sorted_frames[end_idx]
#
#         # Extract the segment data for all joints
#         segment_data = df_3d[(df_3d['frame_id'] >= start_frame) & (df_3d['frame_id'] <= end_frame)].copy()
#         segments.append(segment_data)
#
#         # Optionally save each segment to a separate CSV
#         csv_pattern = f"sub_{subject_id}_motion_{motion_id}_seg_{i+1}.csv"
#         segment_data.to_csv(os.path.join(destination_folder, csv_pattern), index=False)
#
#
#     # print(f"Segmentation complete. Found {len(segments)} segments.")
#     print(f"Segment frames: {[(segments[i]['frame_id'].min(), segments[i]['frame_id'].max()) for i in range(len(segments))]}")



def load_model_with_full_state(filename, num_segments=None, num_signals=None):
    """
    Load model with full state reconstruction from an enhanced save file
    Optionally specify different numbers of segments or signals
    """
    
    # Load the saved data
    saved_data = torch.load(filename, weights_only=False)
    
    # Count the number of weights segments in the saved model
    saved_segments = 0
    for key in saved_data["model_state_dict"]:
        if key.startswith("weights."):
            segment_idx = int(key.split('.')[1])
            saved_segments = max(saved_segments, segment_idx + 1)
    
    # Use provided values or defaults from saved data
    if num_segments is None:
        num_segments = saved_segments
    
    # Get number of signals from first weight parameter
    if num_signals is None and "weights.0" in saved_data["model_state_dict"]:
        first_weight = saved_data["model_state_dict"]["weights.0"]
        num_signals = first_weight.shape[0]
    
    # Create a new model with saved configurations
    model = MP_model(
        num_t_points=saved_data["num_t_points"],
        num_MPs=saved_data["num_MPs"],
        kernel_width=saved_data["kernel_width"],
        kernel_var=saved_data["kernel_var"],
        noise_level=saved_data["noise_level"],
        num_signals=num_signals,
        num_segments=num_segments
    )
    
    # Filter the state dict to only include keys that match the current model
    filtered_state_dict = {}
    for key, value in saved_data["model_state_dict"].items():
        if key.startswith("weights."):
            segment_idx = int(key.split('.')[1])
            if segment_idx < num_segments:
                filtered_state_dict[key] = value
        else:
            filtered_state_dict[key] = value
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # Replace kernel matrices with saved versions
    model.K = saved_data["K"]
    model.invK = saved_data["invK"]
    
    # Restore resampling matrices with correct numeric keys
    model.resampling_matrix = {}
    for k_str, v in saved_data["resampling_matrix"].items():
        model.resampling_matrix[int(k_str)] = v
    
    model.learn_curve = saved_data["learn_curve"]
    model.VAF_curve = saved_data["VAF_curve"] 

    return model


def save_model_with_full_state(model, filename):
    """Save the model with all state information needed for exact reproduction"""
    save_data = {
     
        "model_state_dict": model.state_dict(),
        
        "num_MPs": model.num_MPs,
        "num_t_points": model.num_t_points,
        "noise_level": model.noise_level,
        
        "kernel_width": model.kernel_width,
        "kernel_var": model.kernel_var,
        
        "resampling_matrix": {str(k): v for k, v in model.resampling_matrix.items()},
        
        "K": model.K,
        "invK": model.invK,

        "learn_curve": model.learn_curve if hasattr(model, 'learn_curve') else [],
        "VAF_curve": model.VAF_curve if hasattr(model, 'VAF_curve') else []
    }
    
    # Save all the data
    torch.save(save_data, filename)
    
