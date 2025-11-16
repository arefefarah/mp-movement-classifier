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
from scipy.signal import butter, filtfilt, find_peaks

from mp_movement_classifier.tmp_extraction.TMP_model import MP_model
from mp_movement_classifier.utils import config

# ATTENTION
### values for json to csv functions:
# H36M_KEYPOINT_NAMES = [
#     'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
#     'Spine', 'Thorax', 'Neck', 'Head',
#     'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
# ]
#
# # Define connections between joints for visualization (based on Human3.6M skeleton)
# SKELETON_CONNECTIONS = [
#     (0, 1), (0, 4),       # Hip to RHip, Hip to LHip
#     (1, 2), (2, 3),       # Right leg
#     (4, 5), (5, 6),       # Left leg
#     (0, 7), (7, 8),       # Spine to thorax
#     (8, 9), (9, 10),      # Thorax to head
#     (8, 11), (11, 12), (12, 13),  # Left arm
#     (8, 14), (14, 15), (15, 16)   # Right arm
# ]

#### values for csv to bvh format consistant with final bvh format
H36M_KEYPOINT_NAMES = [
    'Hip', 'RightHip', 'RightKnee', 'RightAnkle', 'LeftHip', 'LeftKnee', 'LeftAnkle',
    'Spine', 'Thorax', 'Neck', 'HeadEndSite', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
    'RightShoulder', 'RightElbow', 'RightWrist'
]

# CORRECTED: Skeleton connections matching the H36M skeleton structure (17 joints)
SKELETON_CONNECTIONS = [
    # Core body connections
    (0, 1), (0, 4),  # Hip -> RightHip, Hip -> LeftHip
    (0, 7),  # Hip -> Spine
    (7, 8),  # Spine -> Thorax
    (8, 9),  # Thorax -> Neck
    (9, 10),  # Neck -> HeadEndSite

    # Right leg
    (1, 2),  # RightHip -> RightKnee
    (2, 3),  # RightKnee -> RightAnkle

    # Left leg
    (4, 5),  # LeftHip -> LeftKnee
    (5, 6),  # LeftKnee -> LeftAnkle

    # Left arm
    (8, 11),  # Thorax -> LeftShoulder
    (11, 12),  # LeftShoulder -> LeftElbow
    (12, 13),  # LeftElbow -> LeftWrist

    # Right arm
    (8, 14),  # Thorax -> RightShoulder
    (14, 15),  # RightShoulder -> RightElbow
    (15, 16),  # RightElbow -> RightWrist
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


def parse_bvh_robust(file_path):
    """
    Robust BVH parser that handles various format issues
    """
    with open(file_path, 'r') as file:
        content = file.read()


    # Split into hierarchy and motion sections
    parts = content.split('MOTION')
    if len(parts) < 2:
        print("❌ Invalid BVH format: No MOTION section found")
        return None, None, None, None

    hierarchy = parts[0]
    motion_part = parts[1]

    # Extract joint information
    joints = {}
    channel_index = 0

    # Find all joints and their channels
    joint_pattern = r'(ROOT|JOINT)\s+(\w+)'
    channel_pattern = r'CHANNELS\s+(\d+)\s+(.*)'

    lines = hierarchy.split('\n')
    current_joint = None

    for line in lines:
        line = line.strip()

        # Find joint names
        joint_match = re.search(joint_pattern, line)
        if joint_match:
            current_joint = joint_match.group(2)

        # Find channels
        channel_match = re.search(channel_pattern, line)
        if channel_match and current_joint:
            num_channels = int(channel_match.group(1))
            channels = channel_match.group(2).split()

            joints[current_joint] = {
                'channels': channels,
                'start_index': channel_index
            }
            channel_index += num_channels

    # Extract motion data
    motion_lines = motion_part.strip().split('\n')

    # Get frame info
    frames = 0
    frame_time = 0.0

    for line in motion_lines:
        if line.startswith('Frames:'):
            frames = int(line.split(':')[1].strip())
        elif line.startswith('Frame Time:'):
            frame_time = float(line.split(':')[1].strip())

    if frames == 0:
        print("❌ No frame information found in BVH file")
        return None, None, None, None

    # Extract frame data
    frame_data = []

    for line in motion_lines:
        line = line.strip()
        if line and not line.startswith('Frames') and not line.startswith('Frame Time'):
            try:
                values = [float(x) for x in line.split()]
                frame_data.extend(values)
            except ValueError:
                continue

    # Calculate expected total channels
    total_channels = sum(len(joint['channels']) for joint in joints.values())
    expected_data_points = total_channels * frames

    if len(frame_data) < expected_data_points:
        print(f"⚠️ Warning: Less data than expected. Using available frames.")
        available_frames = len(frame_data) // total_channels
        motion_data = np.array(frame_data[:available_frames * total_channels]).reshape(available_frames, total_channels)
        frames = available_frames
    else:
        motion_data = np.array(frame_data[:expected_data_points]).reshape(frames, total_channels)

    return joints, motion_data, frame_time, frames

def extract_joint_angles_robust(joints, motion_data, joint_name):
    """
    Extract rotation angles for a specific joint with error handling
    """
    if joint_name not in joints:
        available_joints = list(joints.keys())
        print(f"Joint '{joint_name}' not found.")
        print(f"Available joints: {available_joints}")
        return None

    joint_info = joints[joint_name]
    start_idx = joint_info['start_index']
    channels = joint_info['channels']

    angles = {}
    for i, channel in enumerate(channels):
        if 'rotation' in channel.lower():
            if start_idx + i < motion_data.shape[1]:
                angles[channel] = motion_data[:, start_idx + i]
            else:
                print(f"Channel index out of range for {joint_name}.{channel}")

    return angles if angles else None


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


def filter_motion_data(data, cutoff_freq=6.0, sampling_rate=None, filter_order=4):
    """
    Apply sixth-order zero-lag Butterworth filter to motion capture data

    Args:
        data: motion data array
        cutoff_freq: cutoff frequency in Hz (default 6.0)
        sampling_rate: frames per second (if None, inferred from data length/time)
        filter_order: filter order (default 6)

    Returns:
        filtered_data: filtered motion data
    """
    # If sampling rate not provided, use a default or try to compute
    if sampling_rate is None:
        # Estimate sampling rate (assuming uniform sampling)
        sampling_rate = 1.0 / (data.shape[0] * 0.033)  # Assuming ~30 fps if not specified

    # Compute Nyquist frequency
    nyquist_freq = sampling_rate / 2.0

    # Validate cutoff frequency
    if cutoff_freq >= nyquist_freq:
        print(f"⚠️ Warning: Cutoff frequency ({cutoff_freq} Hz) is too high for sampling rate ({sampling_rate:.1f} Hz)")
        cutoff_freq = nyquist_freq * 0.8  # Use 80% of Nyquist frequency
        print(f"   Adjusting cutoff to {cutoff_freq:.1f} Hz")

    # Normalize cutoff frequency
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Design Butterworth filter
    b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)

    # Apply filter to each channel
    filtered_data = np.zeros_like(data)

    # Suppress warnings for small datasets
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for channel in range(data.shape[1]):
            # Use filtfilt for zero-phase filtering (bidirectional)
            filtered_data[:, channel] = filtfilt(b, a, data[:, channel])

    return filtered_data

def segment_motion_trajectories(motion_data, joints, frame_time,
                                wrist_joints=['LeftWrist', 'RightWrist'],
                                ankle_joints=['LeftAnkle', 'RightAnkle'],
                                min_boundary_distance=0.160):  # 160 ms
    """
    Segment motion trajectories based on joint speed and visualize full joint trajectories

    Returns:
        Tuple of (segments, boundary_frames, joint_speeds)
    """
    print("\n📊 Motion Segments")
    joint_speeds = compute_joint_speed(motion_data, joints, frame_time,
                                       wrist_joints, ankle_joints)
    # print("joint speed:",joint_speeds)
    # Minimum distance in frames
    min_frames = int(min_boundary_distance / frame_time)
    # min_frames = 5 # i manually change it to 5 instead of 4
    # min_frames for the paper was 160 milisecond for 120 Hz means 19 frames

    print(f"Minimum distance in frames: {min_frames}")

    # Find speed minima as potential segment boundaries
    peaks, _ = find_peaks(-joint_speeds, distance=min_frames)
    # print(f"Peaks: {peaks}")

    boundary_frames = [0] + list(peaks) + [len(joint_speeds) - 1]
    # print(f"boundary_frames: {boundary_frames}")
    boundary_frames.sort()

    boundaries = [boundary_frames[i:i + 2] for i in range(len(boundary_frames) - 1)]
    segments = [motion_data[boundary_frames[i]:boundary_frames[i + 1],:] for i in range(len(boundary_frames) - 1)]

    time_vector = np.arange(len(joint_speeds)) * frame_time

    print(f"Duration of complete video : {len(joint_speeds)* frame_time} seconds")
    print(f"Number of segments: {len(segments)}")
    for i, segment in enumerate(segments, 1):
        boundary = boundaries[i - 1]
        print("segment shape",segment.shape)
        print(f"   Segment {i}: Frames {boundary[0]}-{boundary[1]} ")
        print(f"   Time: {time_vector[boundary[0]]} s - {time_vector[boundary[1]]} s")

    return segments, boundaries,boundary_frames, joint_speeds


def process_bvh_data(data_dir, motion_ids, cutoff_freq=6.0):
    """
    Apply Butterworth filter and segmentation to BVH motion data

    Args:
        bvh_data: list of BVH mocap objects
        motion_ids: list of motion IDs corresponding to each BVH sequence
        cutoff_freq: cutoff frequency for filtering (default 6.0 Hz)

    Returns:
        processed_segments: list of segmented motion data
        segment_motion_ids: list of motion IDs corresponding to each segment
    """
    processed_segments = []
    segment_motion_ids = []
    bvh_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.bvh')]

    for file, motion_id in zip(bvh_files, motion_ids):
        file_dir = os.path.join(data_dir, file)
        print(f"Processing {file_dir} with motion ID {motion_id}")
        joints, motion_data, frame_time, frames = parse_bvh_robust(file_dir)

        # Apply Butterworth filter
        # smoothed_motion_data = filter_motion_data(motion_data, cutoff_freq=cutoff_freq,sampling_rate =30)
        smoothed_motion_data = motion_data

        # Apply temporal segmentation
        segments, boundaries,boundary_frames, speeds = segment_motion_trajectories(
            smoothed_motion_data,
            joints,
            frame_time,
            min_boundary_distance = 1 # 1 second
        )

        # print(f"   ✅ Found {len(segments)} motion segments")
        min_segment_length = 10
        for segment in segments:
            if segment.shape[0] >= min_segment_length:

                processed_segments.append(segment.T)  # Transpose to [signals, time]
                segment_motion_ids.append(motion_id)
        #     print("\n******  with segmentation")
        #     print("segment first: ", segment[:,10])

        # pass without segmentation
        # processed_segments.append(smoothed_motion_data.T)# the format of each segment should be [signals,time
        # segment_motion_ids.append(motion_id)
        # print("\n#### without segmentation")
        # print("segment first: ", smoothed_motion_data[:, 10])

    if not processed_segments:
        raise ValueError("No segments could be processed")

    return processed_segments, segment_motion_ids

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
    
