#!/usr/bin/env python3
"""
MoGaze Dataset Preprocessor: Convert Euler angles to 3D positions
Follows the GazeMotion preprocessing (Hu et al., IROS 2024)
"""

import os
import h5py
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def quaternion_matrix(quaternion):
    """Return rotation matrix from quaternion (ROS convention x,y,z,w)"""
    quaternion_tmp = np.array([0.0] * 4)
    quaternion_tmp[1] = quaternion[0]  # x
    quaternion_tmp[2] = quaternion[1]  # y
    quaternion_tmp[3] = quaternion[2]  # z
    quaternion_tmp[0] = quaternion[3]  # w
    q = np.array(quaternion_tmp, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    _EPS = np.finfo(float).eps * 4.0
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])


def euler2xyz(pose_euler):
    """
    Convert human pose from Euler representation to 3D positions.
    Same forward kinematics as the GazeMotion preprocessing.
    
    Args:
        pose_euler: [frames, 66] - Euler angles (22 joints × 3 rotations)
    
    Returns:
        pose_xyz: [frames, 63] - 3D positions (21 joints × 3 positions)
    """
    # Names of all 21 joints (excluding base position which is separate)
    joint_names = ['base', 'pelvis', 'torso', 'neck', 'head', 'linnerShoulder',  
                   'lShoulder', 'lElbow', 'lWrist', 'rinnerShoulder', 'rShoulder', 
                   'rElbow', 'rWrist', 'lHip', 'lKnee', 'lAnkle', 
                   'lToe', 'rHip', 'rKnee', 'rAnkle', 'rToe']                                                                                   
    joint_ids = {name: idx for idx, name in enumerate(joint_names)}
                         
    # Translation vectors for 20 joints (excluding base)
    joint_trans = np.array([[0, 0, 0.074],      # pelvis
                            [0, 0, 0.201],      # torso
                            [0, 0, 0.234],      # neck
                            [0, -0.018, 0.140], # head
                            [0.036, 0, 0.183],  # linnerShoulder
                            [0.153, 0, 0],      # lShoulder
                            [0.243, 0, 0],      # lElbow
                            [0.267, -0.002, 0], # lWrist
                            [-0.036, 0, 0.183], # rinnerShoulder
                            [-0.153, 0, 0],     # rShoulder
                            [-0.243, 0, 0],     # rElbow
                            [-0.267, -0.002, 0], # rWrist
                            [0.090, 0, 0],      # lHip
                            [0, 0, -0.383],     # lKnee
                            [0, 0, -0.354],     # lAnkle
                            [0, -0.135, -0.059], # lToe
                            [-0.090, 0, 0],     # rHip
                            [0, 0, -0.383],     # rKnee
                            [0, 0, -0.354],     # rAnkle
                            [0, -0.135, -0.059]]) # rToe

    # Parent relationships for kinematic chain
    joint_parent_names = {
        'base': 'base',
        'pelvis': 'base',                               
        'torso': 'pelvis', 
        'neck': 'torso', 
        'head': 'neck', 
        'linnerShoulder': 'torso',
        'lShoulder': 'linnerShoulder', 
        'lElbow': 'lShoulder', 
        'lWrist': 'lElbow', 
        'rinnerShoulder': 'torso', 
        'rShoulder': 'rinnerShoulder', 
        'rElbow': 'rShoulder', 
        'rWrist': 'rElbow', 
        'lHip': 'base', 
        'lKnee': 'lHip', 
        'lAnkle': 'lKnee', 
        'lToe': 'lAnkle', 
        'rHip': 'base', 
        'rKnee': 'rHip', 
        'rAnkle': 'rKnee', 
        'rToe': 'rAnkle'
    }                               
    
    joint_parent_ids = [joint_ids[joint_parent_names[child_name]] for child_name in joint_names]
        
    # Forward kinematics computation
    joint_number = len(joint_names)
    pose_xyz = np.zeros((pose_euler.shape[0], joint_number * 3))
    
    for i in range(pose_euler.shape[0]):        
        # Initialize joint positions
        pose_xyz_tmp = np.zeros((joint_number, 3))
        pose_xyz_tmp[0] = [pose_euler[i][0], pose_euler[i][1], pose_euler[i][2]]  # Base position
        
        # Compute rotation matrices for all joints
        pose_rot_mat = np.zeros((joint_number, 3, 3))
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            pose_rot_mat[j] = R.from_euler('XYZ', rot).as_matrix()
                          
        # Apply forward kinematics
        for j in range(1, joint_number):
            pose_rot_mat_parent = pose_rot_mat[joint_parent_ids[j]]
            pose_xyz_tmp[j] = np.matmul(pose_rot_mat_parent, joint_trans[j-1]) + pose_xyz_tmp[joint_parent_ids[j]]
            pose_rot_mat[j] = np.matmul(pose_rot_mat_parent, pose_rot_mat[j])
        
        pose_xyz[i] = pose_xyz_tmp.reshape(joint_number * 3)
    
    return pose_xyz


def preprocess_mogaze_to_3d(input_dir, output_dir):
    """
    Preprocess MoGaze dataset: Convert Euler angles to 3D positions
    
    Args:
        input_dir: Path to original MoGaze HDF5 files
        output_dir: Path to save preprocessed 3D position data
    """
    print("Preprocessing MoGaze dataset: Euler angles → 3D positions")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sessions used (participant 3 excluded)
    data_idx = ["1_1", "1_2", "2_1", "4_1", "5_1", "6_1", "6_2", "7_1", "7_3"]
    
    # Processing parameters
    original_fps = 120.0
    downsample_rate = 4  # 120Hz → 30Hz
    confidence_level_threshold = 0.6
    confidence_ratio_threshold = 0.8
    
    total_sequences = 0
    processed_sequences = 0
    
    for participant in tqdm(data_idx, desc="Processing participants"):
        print(f"\nProcessing participant p{participant}")
        
        # File paths
        pose_file = os.path.join(input_dir, f'p{participant}_human_data.hdf5')
        gaze_file = os.path.join(input_dir, f'p{participant}_gaze_data.hdf5')
        object_file = os.path.join(input_dir, f'p{participant}_object_data.hdf5')
        seg_file = os.path.join(input_dir, f'p{participant}_segmentations.hdf5')
        
        # Check if files exist
        for file_path in [pose_file, gaze_file, object_file, seg_file]:
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping participant")
                continue
        
        # Load data
        try:
            # Load human pose (Euler angles)
            with h5py.File(pose_file, 'r') as f:
                pose_euler = f['data'][:]  # [frames, 66]
            print(f"   Loaded pose data: {pose_euler.shape}")
            
            # Load eye gaze data
            with h5py.File(gaze_file, 'r') as f:
                gaze_raw = f['gaze'][:, 2:5]
                confidence = f['gaze'][:, -1]
                calib = f['gaze'].attrs['calibration']
            
            # Load object data (for gaze calibration)
            with h5py.File(object_file, 'r') as f:
                # Find goggles object for gaze calibration
                goggles_data = None
                for key in f['bodies'].keys():
                    obj_data = f['bodies/' + key][:]
                    # Use the eye-tracker goggles (or the first body with matching length)
                    if 'goggles' in key.lower() or len(obj_data) == len(pose_euler):
                        goggles_data = obj_data
                        break
                
                if goggles_data is None:
                    # Use first object as fallback
                    first_key = list(f['bodies'].keys())[0]
                    goggles_data = f['bodies/' + first_key][:]
            
            # Process gaze data (calibration)
            gaze_processed = np.zeros((pose_euler.shape[0], 3))
            gaze_confidence = confidence
            
            for i in range(len(gaze_processed)):
                rotmat = quaternion_matrix(calib)
                if goggles_data is not None and i < len(goggles_data):
                    rotmat = np.dot(quaternion_matrix(goggles_data[i, 3:7]), rotmat)
                
                endpos = gaze_raw[i]
                if endpos[2] < 0:
                    endpos *= -1
                endpos = np.dot(rotmat, endpos)
                
                if goggles_data is not None and i < len(goggles_data):
                    direction = endpos - goggles_data[i][0:3]
                else:
                    direction = endpos
                
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                gaze_processed[i] = direction
            
            # Load segmentation data
            with h5py.File(seg_file, 'r') as f:
                segments = f['segments'][:]
            
            print(f"   Found {len(segments)-1} action segments")
            
        except Exception as e:
            print(f"Error loading data for p{participant}: {e}")
            continue
        
        # Process each action segment
        participant_sequences = 0
        for i in range(len(segments) - 1):
            total_sequences += 1
            
            current_segment = segments[i]
            next_segment = segments[i + 1]
            
            start_frame = int(current_segment[0])
            end_frame = int(current_segment[1])
            
            # Check sequence length
            if end_frame - start_frame < 10:  # Too short
                continue
            
            # Check gaze quality
            gaze_conf_segment = gaze_confidence[start_frame:end_frame+1]
            high_quality_ratio = np.sum(gaze_conf_segment >= confidence_level_threshold) / len(gaze_conf_segment)
            
            if high_quality_ratio < confidence_ratio_threshold:
                continue
            
            # Extract and downsample data
            pose_euler_segment = pose_euler[start_frame:end_frame+1:downsample_rate, :]
            gaze_segment = gaze_processed[start_frame:end_frame+1:downsample_rate, :]
            
            # Skip if too short after downsampling
            if len(pose_euler_segment) < 10:
                continue
            
            # Convert Euler angles to 3D positions
            try:
                pose_3d_segment = euler2xyz(pose_euler_segment)
                print(f"   Converted segment {i}: {pose_euler_segment.shape} → {pose_3d_segment.shape}")
            except Exception as e:
                print(f"   Error converting segment {i}: {e}")
                continue
            
            # Determine action type
            current_object = current_segment[2].decode("utf-8") if len(current_segment) > 2 else "unknown"
            next_object = next_segment[2].decode("utf-8") if len(next_segment) > 2 else "unknown"
            
            if next_object == "null":
                action = "place"
            else:
                action = "pick"
            
            # Save processed data
            sequence_id = f"p{participant}_{action}_{participant_sequences:03d}"
            
            # Save as .npy files
            np.save(os.path.join(output_dir, f"{sequence_id}_pose_3d.npy"), pose_3d_segment)
            np.save(os.path.join(output_dir, f"{sequence_id}_pose_euler.npy"), pose_euler_segment)
            np.save(os.path.join(output_dir, f"{sequence_id}_gaze.npy"), gaze_segment)
            
            # Save metadata
            metadata = {
                'participant': participant,
                'action': action,
                'sequence_id': participant_sequences,
                'original_frames': (start_frame, end_frame),
                'fps': 30.0,
                'gaze_quality': high_quality_ratio,
                'duration_seconds': len(pose_3d_segment) / 30.0
            }
            np.save(os.path.join(output_dir, f"{sequence_id}_metadata.npy"), metadata)
            
            participant_sequences += 1
            processed_sequences += 1
    
    print(f"\nPreprocessing complete!")
    print(f"   Total sequences found: {total_sequences}")
    print(f"   Successfully processed: {processed_sequences}")
    print(f"   Success rate: {processed_sequences/total_sequences*100:.1f}%")
    print(f"   Output directory: {output_dir}")
    
    # Create summary file
    summary = {
        'total_sequences': total_sequences,
        'processed_sequences': processed_sequences,
        'participants': data_idx,
        'output_format': 'pose_3d: [frames, 63], pose_euler: [frames, 66], gaze: [frames, 3]',
        'fps': 30.0,
        'downsample_rate': downsample_rate
    }
    np.save(os.path.join(output_dir, 'preprocessing_summary.npy'), summary)
    

if __name__ == "__main__":
    import argparse

    # Default paths are relative to this file (the repository root)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Convert raw MoGaze HDF5 files into 3D pose + gaze .npy files")
    parser.add_argument('--input-dir', default=os.path.join(repo_root, 'data', 'mogaze'),
                        help='folder with the original MoGaze HDF5 files')
    parser.add_argument('--output-dir', default=os.path.join(repo_root, 'data', 'mogaze_3d'),
                        help='folder to write the preprocessed .npy files to')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    print("MoGaze 3D Position Preprocessor")
    print("=" * 50)

    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        print("Please pass --input-dir with your MoGaze dataset location")
        exit(1)

    # Run preprocessing
    preprocess_mogaze_to_3d(input_dir, output_dir)