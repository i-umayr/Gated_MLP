import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data

# Leave-one-person-out fold: participant p6 is held out for testing
TRAIN_SESSIONS = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p7_1', 'p7_3']
TEST_SESSIONS = ['p6_1', 'p6_2']


class MoGaze3DDataset(data.Dataset):
    """MoGaze training set of 3D joint positions (21 joints x 3 = 63 dims)."""
    def __init__(self, config, split_name, data_aug=False):
        super(MoGaze3DDataset, self).__init__()
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
            selected = TRAIN_SESSIONS
        else:
            selected = TEST_SESSIONS

        return selected

    def _get_mogaze_files(self):
        """Get MoGaze 3D position files based on participant split"""
        mogaze_files = []

        all_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))

        for participant in self.participant_splits:
            # File names start with the session ID, e.g. p1_1_pick_001_pose_3d.npy
            participant_files = [f for f in all_files if os.path.basename(f).startswith(participant)]
            mogaze_files.extend(participant_files)

        print(f"{self._split_name} split: {len(mogaze_files)} files from sessions {self.participant_splits}")
        return mogaze_files

    def _load_mogaze_file(self, file_path):
        """Load a single MoGaze 3D position file"""
        try:
            motion_data = np.load(file_path)  # [frames, 63]

            frames, total_dims = motion_data.shape

            if total_dims != 63:
                raise ValueError(f"Expected 63 dimensions (21 joints × 3 positions), got {total_dims}")

            return motion_data

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _collect_all(self):
        """Collect all valid motion sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0

        for file_path in tqdm(self._mogaze_files, desc="Loading MoGaze 3D files"):
            motion_data = self._load_mogaze_file(file_path)

            if motion_data is None:
                continue

            N = motion_data.shape[0]

            # Skip sequences that are too short
            min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length
            if N < min_length:
                continue

            self.mogaze_seqs.append(motion_data)

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


class MoGaze3DEval(data.Dataset):
    """MoGaze test set of the held-out participant (a window every 10 frames)."""
    def __init__(self, config, split_name):
        super(MoGaze3DEval, self).__init__()
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
        """Get test files for evaluation"""
        test_files = []
        all_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))

        for participant in self.test_participants:
            participant_files = [f for f in all_files if os.path.basename(f).startswith(participant)]
            test_files.extend(participant_files)

        return test_files

    def _collect_all(self):
        """Collect evaluation sequences"""
        self.mogaze_seqs = []
        self.data_idx = []
        idx = 0

        for file_path in self._mogaze_files:
            try:
                motion_data = np.load(file_path)  # [frames, 63]

                N = motion_data.shape[0]
                min_length = self.mogaze_motion_input_length + self.mogaze_motion_target_length

                if N >= min_length:
                    self.mogaze_seqs.append(motion_data)

                    # A new test window starts every 10 frames
                    valid_frames = np.arange(0, N - min_length + 1, 10)
                    self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                    idx += 1

            except Exception as e:
                print(f"Error loading evaluation file {file_path}: {e}")
                continue

        print(f"Test sessions {self.test_participants}: {len(self.mogaze_seqs)} sequences, {len(self.data_idx)} samples")

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]

        end_frame = start_frame + self.mogaze_motion_input_length + self.mogaze_motion_target_length
        motion = self.mogaze_seqs[idx][start_frame:end_frame]

        mogaze_motion_input = motion[:self.mogaze_motion_input_length]
        mogaze_motion_target = motion[self.mogaze_motion_input_length:]

        return torch.tensor(mogaze_motion_input).float(), torch.tensor(mogaze_motion_target).float()
