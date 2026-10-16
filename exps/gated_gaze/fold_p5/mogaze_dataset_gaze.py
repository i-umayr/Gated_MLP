import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data

# Leave-one-person-out fold: participant p5 is held out for testing
TRAIN_SESSIONS = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
TEST_SESSIONS = ['p5_1']


class MoGazeGazeDataset(data.Dataset):
    """MoGaze training set: 3D joint positions (63 dims) + gaze direction (3 dims) = 66 dims."""
    def __init__(self, config, split_name, data_aug=False):
        super(MoGazeGazeDataset, self).__init__()
        self._split_name = split_name
        self.data_aug = data_aug

        self._mogaze_anno_dir = config.mogaze_anno_dir

        self.mogaze_motion_input_length = config.motion.mogaze_input_length

        if split_name == 'train':
            self.mogaze_motion_target_length = config.motion.mogaze_target_length_train
        else:
            self.mogaze_motion_target_length = config.motion.mogaze_target_length_eval

        self.motion_dim = config.motion.dim

        self.shift_step = config.shift_step

        self.participant_splits = self._get_participant_splits()
        self._mogaze_files = self._get_mogaze_files()

        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        return self._file_length

    def _get_participant_splits(self):
        """Sessions used for this split of the fold."""
        if self._split_name == 'train':
            return TRAIN_SESSIONS
        else:
            return TEST_SESSIONS

    def _get_mogaze_files(self):
        """Get MoGaze pose and gaze file pairs"""
        mogaze_file_pairs = []

        all_pose_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))

        for participant in self.participant_splits:
            participant_pose_files = [f for f in all_pose_files
                                    if os.path.basename(f).startswith(participant)]

            # Keep pose files that have a matching gaze file
            valid_pairs = 0
            for pose_file in participant_pose_files:
                gaze_file = pose_file.replace('_pose_3d.npy', '_gaze.npy')
                if os.path.exists(gaze_file):
                    mogaze_file_pairs.append((pose_file, gaze_file))
                    valid_pairs += 1

        print(f"{self._split_name} split: {len(mogaze_file_pairs)} pose-gaze pairs from sessions {self.participant_splits}")
        return mogaze_file_pairs

    def _load_mogaze_file_pair(self, pose_file, gaze_file):
        """Load a pose-gaze file pair and concatenate them"""
        try:
            pose_data = np.load(pose_file)  # [frames, 63]
            gaze_data = np.load(gaze_file)  # [frames, 3]

            frames_pose, pose_dims = pose_data.shape
            frames_gaze, gaze_dims = gaze_data.shape

            if pose_dims != 63:
                raise ValueError(f"Expected 63 pose dimensions, got {pose_dims}")
            if gaze_dims != 3:
                raise ValueError(f"Expected 3 gaze dimensions, got {gaze_dims}")
            if frames_pose != frames_gaze:
                raise ValueError(f"Frame mismatch: pose {frames_pose} vs gaze {frames_gaze}")

            combined_data = np.concatenate([pose_data, gaze_data], axis=1)  # [frames, 66]

            return combined_data

        except Exception as e:
            print(f"Error loading {pose_file} + {gaze_file}: {e}")
            return None

    def _collect_all(self):
        """Collect all valid motion sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0

        for pose_file, gaze_file in tqdm(self._mogaze_files, desc="Loading MoGaze pose+gaze files"):
            combined_data = self._load_mogaze_file_pair(pose_file, gaze_file)

            if combined_data is None:
                continue

            N = combined_data.shape[0]

            # Skip sequences that are too short
            min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length
            if N < min_length:
                continue

            self.mogaze_seqs.append(combined_data)

            valid_frames = np.arange(0, N - min_length + 1, self.shift_step)
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1

        print(f"Collected {len(self.mogaze_seqs)} sequences with {len(self.data_idx)} samples")

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]

        total_frames = self.mogaze_motion_input_length + self.mogaze_motion_target_length
        end_frame = start_frame + total_frames
        motion = self.mogaze_seqs[idx][start_frame:end_frame]

        if self.data_aug:
            if torch.rand(1)[0] > 0.5:
                # Reverse the sequence in time (front-back flip)
                idx_reverse = [i for i in range(motion.shape[0]-1, -1, -1)]
                motion = motion[idx_reverse]

        # Positions are in metres
        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]

        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()


class MoGazeGazeEval(data.Dataset):
    """MoGaze test set of the held-out participant (a window every 10 frames)."""
    def __init__(self, config, split_name):
        super(MoGazeGazeEval, self).__init__()
        self._split_name = split_name
        self._mogaze_anno_dir = config.mogaze_anno_dir

        self.mogaze_motion_input_length = config.motion.mogaze_input_length
        self.mogaze_motion_target_length = config.motion.mogaze_target_length_eval

        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step

        self.test_participants = TEST_SESSIONS
        self._mogaze_files = self._get_test_files()
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        return self._file_length

    def _get_test_files(self):
        """Get test file pairs for evaluation"""
        test_file_pairs = []
        all_pose_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))

        for participant in self.test_participants:
            participant_pose_files = [f for f in all_pose_files
                                    if os.path.basename(f).startswith(participant)]

            valid_pairs = 0
            for pose_file in participant_pose_files:
                gaze_file = pose_file.replace('_pose_3d.npy', '_gaze.npy')
                if os.path.exists(gaze_file):
                    test_file_pairs.append((pose_file, gaze_file))
                    valid_pairs += 1

        return test_file_pairs

    def _collect_all(self):
        """Collect evaluation sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0

        for pose_file, gaze_file in self._mogaze_files:
            try:
                pose_data = np.load(pose_file)  # [frames, 63]
                gaze_data = np.load(gaze_file)  # [frames, 3]

                if pose_data.shape[0] != gaze_data.shape[0]:
                    continue

                combined_data = np.concatenate([pose_data, gaze_data], axis=1)  # [frames, 66]

                N = combined_data.shape[0]
                min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length

                if N >= min_length:
                    self.mogaze_seqs.append(combined_data)

                    # A new test window starts every 10 frames
                    valid_frames = np.arange(0, N - min_length + 1, 10)
                    self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                    idx += 1

            except Exception as e:
                print(f"Error loading evaluation file pair: {e}")
                continue

        print(f"Test sessions {self.test_participants}: {len(self.mogaze_seqs)} sequences, {len(self.data_idx)} samples")

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]

        end_frame = start_frame + self.mogaze_motion_input_length + self.mogaze_motion_target_length
        motion = self.mogaze_seqs[idx][start_frame:end_frame]

        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]

        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()
