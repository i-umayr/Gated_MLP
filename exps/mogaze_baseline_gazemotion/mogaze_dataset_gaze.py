import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data


class MoGazeGazeDataset(data.Dataset):
    """
    MoGaze dataset loader for 3D position + gaze integration with DENSE SAMPLING
    Uses GazeMotion's approach: create all possible overlapping sequences
    """
    def __init__(self, config, split_name, data_aug=False):
        super(MoGazeGazeDataset, self).__init__()
        self._split_name = split_name
        self.data_aug = data_aug
        
        self._mogaze_anno_dir = config.mogaze_anno_dir
        
        # Use the same config names as before for model compatibility
        self.mogaze_motion_input_length = config.motion.mogaze_input_length  # 50 frames
        
        if split_name == 'train':
            self.mogaze_motion_target_length = config.motion.mogaze_target_length_train  # 10 frames
        else:
            self.mogaze_motion_target_length = config.motion.mogaze_target_length_eval   # 30 frames
        
        # Total sequence length needed
        self.seq_len = self.mogaze_motion_input_length + self.mogaze_motion_target_length
        
        self.motion_dim = config.motion.dim  # Should be 66 for pose+gaze
        
        # Dense sampling parameters (following GazeMotion approach)
        self.sample_rate = getattr(config, 'sample_rate', 1)  # Default to 1 (no downsampling)
        
        # Get participant splits (same as before - NO data leakage)
        self.participant_splits = self._get_participant_splits()
        self._mogaze_files = self._get_mogaze_files()
        
        self._collect_all_dense()
        self._file_length = len(self.data_idx)
    
    def __len__(self):
        return self._file_length
    
    def _get_participant_splits(self):
        """Define participant splits - SAME AS BEFORE (no data leakage)"""
        participants = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        
        if self._split_name == 'train':
            return participants[:-2]  # ['p1_1', ..., 'p6_2'] - same as before
        else:
            return participants[-2:]  # ['p7_1', 'p7_3'] - same as before
    
    def _get_mogaze_files(self):
        """Get MoGaze pose and gaze file pairs"""
        mogaze_file_pairs = []
        
        # Look for pose files first
        all_pose_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))
        print(f"🔍 Found {len(all_pose_files)} total pose files")
        
        for participant in self.participant_splits:
            # Match pose files for this participant
            participant_pose_files = [f for f in all_pose_files 
                                    if os.path.basename(f).startswith(participant)]
            
            # For each pose file, check if corresponding gaze file exists
            valid_pairs = 0
            for pose_file in participant_pose_files:
                gaze_file = pose_file.replace('_pose_3d.npy', '_gaze.npy')
                if os.path.exists(gaze_file):
                    mogaze_file_pairs.append((pose_file, gaze_file))
                    valid_pairs += 1
            
            print(f"   Participant {participant}: {valid_pairs} valid pose-gaze pairs")
        
        print(f"✅ Found {len(mogaze_file_pairs)} total pose-gaze pairs for {self._split_name} split")
        print(f"   Participants: {self.participant_splits}")
        return mogaze_file_pairs
    
    def _load_mogaze_file_pair(self, pose_file, gaze_file):
        """Load a pose-gaze file pair and concatenate them"""
        try:
            # Load 3D position data
            pose_data = np.load(pose_file)  # [frames, 63] - 21 joints × 3 positions
            
            # Load gaze data  
            gaze_data = np.load(gaze_file)  # [frames, 3] - 3D gaze direction vectors
            
            frames_pose, pose_dims = pose_data.shape
            frames_gaze, gaze_dims = gaze_data.shape
            
            # Verify dimensions
            if pose_dims != 63:
                raise ValueError(f"Expected 63 pose dimensions, got {pose_dims}")
            if gaze_dims != 3:
                raise ValueError(f"Expected 3 gaze dimensions, got {gaze_dims}")
            if frames_pose != frames_gaze:
                raise ValueError(f"Frame mismatch: pose {frames_pose} vs gaze {frames_gaze}")
            
            # Concatenate pose + gaze = 66D
            combined_data = np.concatenate([pose_data, gaze_data], axis=1)  # [frames, 66]
            
            return combined_data
                
        except Exception as e:
            print(f"Error loading {pose_file} + {gaze_file}: {e}")
            return None
    
    def _collect_all_dense(self):
        """
        Collect all valid motion sequences using DENSE SAMPLING approach
        Following GazeMotion's methodology for fair comparison
        """
        self.mogaze_seqs = []
        self.data_idx = []
        total_samples = 0
        valid_seq_idx = 0  # Track actual sequence index in mogaze_seqs
        
        print(f"🔄 Loading data with DENSE SAMPLING (seq_len={self.seq_len})...")
        
        for file_idx, (pose_file, gaze_file) in enumerate(tqdm(self._mogaze_files, desc="Processing files")):
            combined_data = self._load_mogaze_file_pair(pose_file, gaze_file)
            
            if combined_data is None:
                continue
            
            N = combined_data.shape[0]
            
            # Skip sequences that are too short
            if N < self.seq_len:
                continue
            
            # Store the sequence data
            self.mogaze_seqs.append(combined_data)
            
            # DENSE SAMPLING: Create all possible overlapping sequences
            # This is the key difference from the original implementation
            
            # Create starting frame indices for all possible sequences
            fs = np.arange(0, N - self.seq_len + 1)
            
            # Debug: Validate frame indices are valid
            if len(fs) > 0:
                max_start_frame = fs[-1]
                max_end_frame = max_start_frame + self.seq_len - 1
                if max_end_frame >= N:
                    print(f"WARNING: Invalid frame calculation for seq {valid_seq_idx}")
                    print(f"  N={N}, seq_len={self.seq_len}, max_start={max_start_frame}, max_end={max_end_frame}")
                    # Recalculate to be safe
                    fs = np.arange(0, max(0, N - self.seq_len + 1))
            
            # Apply sample rate (downsample sequences if needed)
            if self.sample_rate > 1:
                fs = fs[::self.sample_rate]
            
            # Store indices: (valid_sequence_idx, start_frame) for each sample
            num_samples = len(fs)
            for start_frame in fs:
                # Double-check each start frame
                if start_frame + self.seq_len > N:
                    print(f"ERROR: Invalid start_frame {start_frame} for seq_len {self.seq_len} and N {N}")
                    continue
                self.data_idx.append((valid_seq_idx, start_frame))
            
            total_samples += num_samples
            valid_seq_idx += 1  # Increment only for valid sequences
            
        print(f"📊 DENSE SAMPLING Results:")
        print(f"   Loaded {len(self.mogaze_seqs)} sequence files")
        print(f"   Generated {len(self.data_idx)} total samples")
        print(f"   Average samples per sequence: {len(self.data_idx) / max(len(self.mogaze_seqs), 1):.1f}")
        print(f"   Data format: [samples, seq_len={self.seq_len}, features=66] (63 pose + 3 gaze)")
        
        if len(self.mogaze_seqs) > 0:
            print(f"   Example sequence shape: {self.mogaze_seqs[0].shape}")
    
    def __getitem__(self, index):
        seq_idx, start_frame = self.data_idx[index]
        
        # Get the sequence from the stored data using start frame
        end_frame = start_frame + self.seq_len
        
        # Ensure we don't go beyond the sequence bounds
        if end_frame > self.mogaze_seqs[seq_idx].shape[0]:
            # This shouldn't happen if dense sampling is correct, but add safety check
            end_frame = self.mogaze_seqs[seq_idx].shape[0]
            start_frame = end_frame - self.seq_len
            
        motion = self.mogaze_seqs[seq_idx][start_frame:end_frame]  # [seq_len, 66]
        
        # Validate sequence length
        if motion.shape[0] != self.seq_len:
            raise ValueError(f"Invalid sequence length: expected {self.seq_len}, got {motion.shape[0]} at index {index}")
        
        # Apply data augmentation if enabled
        if self.data_aug:
            if torch.rand(1)[0] > 0.5:
                # Reverse the sequence (front-back flip)
                motion = motion[::-1].copy()  # Make a copy to fix negative strides
        
        # Split into input and target frames
        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]
        
        # Validate split lengths
        assert mogaze_motion_input.shape[0] == self.mogaze_motion_input_length, f"Input length mismatch: {mogaze_motion_input.shape[0]} vs {self.mogaze_motion_input_length}"
        assert mogaze_motion_target.shape[0] == self.mogaze_motion_target_length, f"Target length mismatch: {mogaze_motion_target.shape[0]} vs {self.mogaze_motion_target_length}"
        
        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()


class MoGazeGazeEval(data.Dataset):
    """Evaluation dataset for MoGaze 3D positions + gaze with DENSE SAMPLING"""
    def __init__(self, config, split_name):
        super(MoGazeGazeEval, self).__init__()
        self._split_name = split_name
        self._mogaze_anno_dir = config.mogaze_anno_dir
        
        self.mogaze_motion_input_length = config.motion.mogaze_input_length
        self.mogaze_motion_target_length = config.motion.mogaze_target_length_eval  # Should be 30 for eval
        self.seq_len = self.mogaze_motion_input_length + self.mogaze_motion_target_length
        
        self.motion_dim = config.motion.dim  # 66 for pose+gaze
        
        # SAME AS TRAINING: Use only p7_1, p7_3 for testing (NO data leakage)
        self.test_participants = ['p7_1', 'p7_3']
        self._mogaze_files = self._get_test_files()
        self._collect_all_dense()
        self._file_length = len(self.data_idx)
    
    def __len__(self):
        return self._file_length
    
    def _get_test_files(self):
        """Get test file pairs for evaluation"""
        test_file_pairs = []
        all_pose_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))
        
        print(f"🔍 Debug eval: Found {len(all_pose_files)} total files")
        
        for participant in self.test_participants:
            # Match pose files for this participant
            participant_pose_files = [f for f in all_pose_files 
                                    if os.path.basename(f).startswith(participant)]
            
            # Check for corresponding gaze files
            valid_pairs = 0
            for pose_file in participant_pose_files:
                gaze_file = pose_file.replace('_pose_3d.npy', '_gaze.npy')
                if os.path.exists(gaze_file):
                    test_file_pairs.append((pose_file, gaze_file))
                    valid_pairs += 1
            
            print(f"   Eval participant {participant}: {valid_pairs} files")
        
        print(f"✅ Eval: Found {len(test_file_pairs)} pose-gaze pairs")
        return test_file_pairs
    
    def _collect_all_dense(self):
        """Collect evaluation sequences with DENSE SAMPLING"""
        self.mogaze_seqs = []
        self.data_idx = []
        
        print(f"🔄 Loading eval data with DENSE SAMPLING...")
        
        for seq_idx, (pose_file, gaze_file) in enumerate(self._mogaze_files):
            try:
                # Load and combine pose + gaze
                pose_data = np.load(pose_file)  # [frames, 63]
                gaze_data = np.load(gaze_file)  # [frames, 3]
                
                if pose_data.shape[0] != gaze_data.shape[0]:
                    continue
                
                combined_data = np.concatenate([pose_data, gaze_data], axis=1)  # [frames, 66]
                
                N = combined_data.shape[0]
                
                if N >= self.seq_len:
                    self.mogaze_seqs.append(combined_data)
                    
                    # DENSE SAMPLING: Create all possible sequences
                    fs = np.arange(0, N - self.seq_len + 1)
                    
                    # For eval, we can optionally use a larger step to reduce computation
                    # but keep dense for fair comparison
                    eval_step = getattr(self, 'eval_step', 1)  # Could be 5 or 10 for faster eval
                    if eval_step > 1:
                        fs = fs[::eval_step]
                    
                    # Store indices
                    num_samples = len(fs)
                    for start_frame in fs:
                        self.data_idx.append((seq_idx, start_frame))
                    
            except Exception as e:
                print(f"Error loading evaluation file pair: {e}")
                continue
        
        print(f"📊 DENSE EVAL Results: {len(self.data_idx)} samples from {len(self.mogaze_seqs)} sequences")
    
    def __getitem__(self, index):
        seq_idx, start_frame = self.data_idx[index]
        
        # Validate seq_idx is within bounds
        if seq_idx >= len(self.mogaze_seqs):
            raise IndexError(f"seq_idx {seq_idx} is out of range for {len(self.mogaze_seqs)} sequences at index {index}")
        
        # Get the sequence from stored data using start frame
        sequence_data = self.mogaze_seqs[seq_idx]
        end_frame = start_frame + self.seq_len
        
        # Ensure we don't go beyond the sequence bounds
        if end_frame > sequence_data.shape[0]:
            # This shouldn't happen if dense sampling is correct, but add safety check
            print(f"WARNING: end_frame {end_frame} > sequence length {sequence_data.shape[0]} for seq_idx {seq_idx}")
            end_frame = sequence_data.shape[0]
            start_frame = max(0, end_frame - self.seq_len)
            
        motion = sequence_data[start_frame:end_frame]  # [seq_len, 66]
        
        # Validate sequence length
        if motion.shape[0] != self.seq_len:
            raise ValueError(f"Invalid eval sequence length: expected {self.seq_len}, got {motion.shape[0]} at index {index}")
        
        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]
        
        # Validate split lengths
        assert mogaze_motion_input.shape[0] == self.mogaze_motion_input_length
        assert mogaze_motion_target.shape[0] == self.mogaze_motion_target_length
        
        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()