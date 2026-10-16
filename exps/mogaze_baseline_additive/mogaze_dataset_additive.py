"""
MoGaze Dataset Loader with LOPOCV Fold Selection (Additive Fusion Ablation)

Identical data handling to the concatenation ablation: loads 3D joint positions
(21 joints x 3 = 63D) + gaze vectors (3D) = 66D total. The additive model splits
pose/gaze internally, so the dataset is unchanged from concat.

Supports parameterized Leave-One-Person-Out Cross-Validation:
  - Specify which participant to hold out for testing
  - Train on the remaining 5 participants

Participant mapping (from MoGaze dataset):
  p1 -> sessions p1_1, p1_2
  p2 -> session  p2_1
  p4 -> session  p4_1
  p5 -> session  p5_1
  p6 -> sessions p6_1, p6_2
  p7 -> sessions p7_1, p7_3
  (p3 excluded due to eye-tracking calibration issues)
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data


# Mapping from participant ID to session prefixes in the dataset files
PARTICIPANT_SESSIONS = {
    'p1': ['p1_1', 'p1_2'],
    'p2': ['p2_1'],
    'p4': ['p4_1'],
    'p5': ['p5_1'],
    'p6': ['p6_1', 'p6_2'],
    'p7': ['p7_1', 'p7_3'],
}

ALL_PARTICIPANTS = ['p1', 'p2', 'p4', 'p5', 'p6', 'p7']


def get_sessions_for_participants(participant_list):
    """Convert participant IDs to session prefixes."""
    sessions = []
    for p in participant_list:
        if p in PARTICIPANT_SESSIONS:
            sessions.extend(PARTICIPANT_SESSIONS[p])
        else:
            raise ValueError(f"Unknown participant: {p}. Valid: {list(PARTICIPANT_SESSIONS.keys())}")
    return sessions


class MoGazeAdditiveDataset(data.Dataset):
    """
    Training dataset for additive fusion ablation.

    Args:
        config: Configuration object
        test_participant: str -- participant held out for testing (e.g., 'p1')
        data_aug: bool -- whether to apply front-back flip augmentation
    """
    def __init__(self, config, test_participant, data_aug=True):
        super(MoGazeAdditiveDataset, self).__init__()
        self.data_aug = data_aug
        self._mogaze_anno_dir = config.mogaze_anno_dir
        self.input_length = config.motion.mogaze_input_length          # 50
        self.target_length = config.motion.mogaze_target_length_train  # 10
        self.shift_step = config.shift_step

        # Determine train participants (all except test_participant)
        self.test_participant = test_participant
        self.train_participants = [p for p in ALL_PARTICIPANTS if p != test_participant]
        self.train_sessions = get_sessions_for_participants(self.train_participants)

        print(f"[LOPOCV] Test participant: {test_participant}")
        print(f"[LOPOCV] Train participants: {self.train_participants}")
        print(f"[LOPOCV] Train sessions: {self.train_sessions}")

        self._mogaze_files = self._get_file_pairs(self.train_sessions)
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        return self._file_length

    def _get_file_pairs(self, sessions):
        """Get pose-gaze file pairs for specified sessions."""
        file_pairs = []
        all_pose_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))

        for session in sessions:
            session_files = [f for f in all_pose_files
                             if os.path.basename(f).startswith(session)]
            valid = 0
            for pose_file in session_files:
                gaze_file = pose_file.replace('_pose_3d.npy', '_gaze.npy')
                if os.path.exists(gaze_file):
                    file_pairs.append((pose_file, gaze_file))
                    valid += 1
            print(f"  Session {session}: {valid} pose-gaze pairs")

        print(f"  Total: {len(file_pairs)} file pairs")
        return file_pairs

    def _collect_all(self):
        """Collect all valid motion sequences with overlapping windows (shift_step)."""
        self.seqs = []
        self.data_idx = []
        idx = 0

        for pose_file, gaze_file in tqdm(self._mogaze_files, desc="Loading train data"):
            try:
                pose_data = np.load(pose_file)   # [frames, 63]
                gaze_data = np.load(gaze_file)    # [frames, 3]

                if pose_data.shape[1] != 63 or gaze_data.shape[1] != 3:
                    continue
                if pose_data.shape[0] != gaze_data.shape[0]:
                    continue

                combined = np.concatenate([pose_data, gaze_data], axis=1)  # [frames, 66]
                N = combined.shape[0]
                min_len = self.input_length + self.target_length

                if N < min_len:
                    continue

                self.seqs.append(combined)
                valid_frames = np.arange(0, N - min_len + 1, self.shift_step)
                self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                idx += 1

            except Exception as e:
                print(f"Error loading {pose_file}: {e}")
                continue

        print(f"Collected {len(self.seqs)} sequences, {len(self.data_idx)} samples")

    def __getitem__(self, index):
        idx, start = self.data_idx[index]
        total = self.input_length + self.target_length
        motion = self.seqs[idx][start:start + total]

        # Front-back flip augmentation
        if self.data_aug and torch.rand(1).item() > 0.5:
            motion = motion[::-1].copy()

        motion_input = motion[:self.input_length]
        motion_target = motion[self.input_length:]

        return torch.tensor(motion_input).float(), torch.tensor(motion_target).float()


class MoGazeAdditiveEval(data.Dataset):
    """
    Evaluation dataset for a single LOPOCV fold.

    Args:
        config: Configuration object
        test_participant: str -- participant held out for testing (e.g., 'p1')
    """
    def __init__(self, config, test_participant):
        super(MoGazeAdditiveEval, self).__init__()
        self._mogaze_anno_dir = config.mogaze_anno_dir
        self.input_length = config.motion.mogaze_input_length          # 50
        self.target_length = config.motion.mogaze_target_length_eval   # 30
        self.shift_step = config.shift_step

        self.test_participant = test_participant
        self.test_sessions = get_sessions_for_participants([test_participant])

        print(f"[LOPOCV Eval] Test participant: {test_participant}")
        print(f"[LOPOCV Eval] Test sessions: {self.test_sessions}")

        self._mogaze_files = self._get_file_pairs(self.test_sessions)
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        return self._file_length

    def _get_file_pairs(self, sessions):
        """Get pose-gaze file pairs for specified sessions."""
        file_pairs = []
        all_pose_files = glob.glob(os.path.join(self._mogaze_anno_dir, "*_pose_3d.npy"))

        for session in sessions:
            session_files = [f for f in all_pose_files
                             if os.path.basename(f).startswith(session)]
            valid = 0
            for pose_file in session_files:
                gaze_file = pose_file.replace('_pose_3d.npy', '_gaze.npy')
                if os.path.exists(gaze_file):
                    file_pairs.append((pose_file, gaze_file))
                    valid += 1
            print(f"  Eval session {session}: {valid} files")

        print(f"  Eval total: {len(file_pairs)} file pairs")
        return file_pairs

    def _collect_all(self):
        """Collect evaluation sequences (stride 10 -- matches original eval protocol)."""
        self.seqs = []
        self.data_idx = []
        idx = 0

        for pose_file, gaze_file in self._mogaze_files:
            try:
                pose_data = np.load(pose_file)
                gaze_data = np.load(gaze_file)

                if pose_data.shape[0] != gaze_data.shape[0]:
                    continue

                combined = np.concatenate([pose_data, gaze_data], axis=1)
                N = combined.shape[0]
                min_len = self.input_length + self.target_length

                if N >= min_len:
                    self.seqs.append(combined)
                    # Sampling every 10 frames -- matches original eval protocol
                    valid_frames = np.arange(0, N - min_len + 1, 10)
                    self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                    idx += 1

            except Exception as e:
                print(f"Error loading eval file: {e}")
                continue

        print(f"Eval: {len(self.seqs)} sequences, {len(self.data_idx)} samples")

    def __getitem__(self, index):
        idx, start = self.data_idx[index]
        total = self.input_length + self.target_length
        motion = self.seqs[idx][start:start + total]

        motion_input = motion[:self.input_length]
        motion_target = motion[self.input_length:]

        return torch.tensor(motion_input).float(), torch.tensor(motion_target).float()


if __name__ == "__main__":
    from mogaze_config_additive import config

    print("Testing LOPOCV dataset splits (additive)...")
    for test_p in ALL_PARTICIPANTS:
        print(f"\n{'='*60}")
        print(f"Fold: test on {test_p}")
        train_ds = MoGazeAdditiveDataset(config, test_participant=test_p, data_aug=False)
        eval_ds = MoGazeAdditiveEval(config, test_participant=test_p)
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Eval samples:  {len(eval_ds)}")
