import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

class GazeMotionDatasetAnalyzer:
    """Analyzer for GazeMotion preprocessed dataset structure"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.participant_data = {}
        
    def load_participant_data(self, participant):
        """Load data for a specific participant"""
        participant_dir = self.data_dir / participant
        if not participant_dir.exists():
            print(f"Participant directory not found: {participant_dir}")
            return None
            
        data_info = {
            'participant': participant,
            'actions': defaultdict(list),
            'total_sequences': 0,
            'total_samples': 0,
            'file_info': []
        }
        
        # Find all files for this participant
        pose_euler_files = list(participant_dir.glob("*_pose_euler.npy"))
        pose_xyz_files = list(participant_dir.glob("*_pose_xyz.npy"))
        gaze_files = list(participant_dir.glob("*_gaze.npy"))
        head_files = list(participant_dir.glob("*_head.npy"))
        
        print(f"\nParticipant {participant}:")
        print(f"  Pose euler files: {len(pose_euler_files)}")
        print(f"  Pose XYZ files: {len(pose_xyz_files)}")
        print(f"  Gaze files: {len(gaze_files)}")
        print(f"  Head files: {len(head_files)}")
        
        # Analyze each sequence
        for pose_file in pose_euler_files:
            try:
                # Parse filename: pick_001_pose_euler.npy or place_001_pose_euler.npy
                filename = pose_file.stem
                parts = filename.split('_')
                if len(parts) >= 3:
                    action = parts[0]  # pick or place
                    sequence_num = parts[1]  # 001, 002, etc.
                    
                    # Load pose data
                    pose_data = np.load(pose_file)
                    
                    # Find corresponding files
                    base_name = f"{action}_{sequence_num}"
                    xyz_file = participant_dir / f"{base_name}_pose_xyz.npy"
                    gaze_file = participant_dir / f"{base_name}_gaze.npy"
                    head_file = participant_dir / f"{base_name}_head.npy"
                    
                    xyz_data = None
                    gaze_data = None
                    head_data = None
                    
                    if xyz_file.exists():
                        xyz_data = np.load(xyz_file)
                    if gaze_file.exists():
                        gaze_data = np.load(gaze_file)
                    if head_file.exists():
                        head_data = np.load(head_file)
                    
                    sequence_info = {
                        'action': action,
                        'sequence_num': sequence_num,
                        'pose_euler_shape': pose_data.shape,
                        'pose_xyz_shape': xyz_data.shape if xyz_data is not None else None,
                        'gaze_shape': gaze_data.shape if gaze_data is not None else None,
                        'head_shape': head_data.shape if head_data is not None else None,
                        'duration_frames': pose_data.shape[0],
                        'duration_seconds': pose_data.shape[0] / 30.0,
                        'filename': filename
                    }
                    
                    data_info['actions'][action].append(sequence_info)
                    data_info['total_sequences'] += 1
                    data_info['file_info'].append(sequence_info)
                    
                    # Calculate potential sliding window samples
                    window_size = 40  # input + output frames
                    stride = 2  # based on train_sample_rate=2
                    if pose_data.shape[0] >= window_size:
                        samples = (pose_data.shape[0] - window_size) // stride + 1
                        data_info['total_samples'] += samples
                    
            except Exception as e:
                print(f"Error loading {pose_file}: {e}")
                
        return data_info
    
    def analyze_all_participants(self):
        """Analyze all participants in the dataset"""
        print("Analyzing GazeMotion preprocessed dataset...")
        print("=" * 60)
        
        # List all participant directories
        participant_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('p')]
        participant_names = sorted([d.name for d in participant_dirs])
        
        print(f"Found participants: {participant_names}")
        
        all_data = {}
        total_sequences = 0
        total_estimated_samples = 0
        
        for participant in participant_names:
            data = self.load_participant_data(participant)
            if data:
                all_data[participant] = data
                total_sequences += data['total_sequences']
                total_estimated_samples += data['total_samples']
        
        return all_data, total_sequences, total_estimated_samples
    
    def compare_sampling_strategies(self, all_data):
        """Compare different sampling strategies"""
        print("\n" + "=" * 80)
        print("SAMPLING STRATEGY ANALYSIS")
        print("=" * 80)
        
        # Focus on test participants (p7_1, p7_3) to match your comparison
        test_participants = ['p7_1', 'p7_3']
        test_sequences = 0
        test_samples_estimated = 0
        test_total_frames = 0
        
        for participant in test_participants:
            if participant in all_data:
                data = all_data[participant]
                test_sequences += data['total_sequences']
                test_samples_estimated += data['total_samples']
                
                # Calculate total frames
                for seq_info in data['file_info']:
                    test_total_frames += seq_info['duration_frames']
        
        print(f"Test Participants Analysis (p7_1, p7_3):")
        print(f"  Raw sequences: {test_sequences}")
        print(f"  Total frames: {test_total_frames:,}")
        print(f"  Estimated sliding window samples: {test_samples_estimated:,}")
        print(f"  Actual GazeMotion samples: 22,901")
        print(f"  Difference: {abs(test_samples_estimated - 22901):,}")
        
        if abs(test_samples_estimated - 22901) < 2000:
            print("  ✓ Sliding window estimation matches GazeMotion count!")
        else:
            print("  ⚠ Significant difference - other factors involved")
            
        # Compare with your dataset
        your_test_sequences = 504  # From your analysis
        
        print(f"\nComparison with Your Dataset:")
        print(f"  Your test sequences: {your_test_sequences}")
        print(f"  GazeMotion raw sequences: {test_sequences}")
        print(f"  GazeMotion sliding samples: 22,901")
        print(f"  Sample inflation ratio: {22901 / your_test_sequences:.1f}x")
        
        return test_sequences, test_samples_estimated
    
    def analyze_sequence_characteristics(self, all_data):
        """Analyze sequence characteristics across participants"""
        print("\n" + "=" * 60)
        print("SEQUENCE CHARACTERISTICS")
        print("=" * 60)
        
        all_durations = []
        action_counts = defaultdict(int)
        participant_stats = []
        
        for participant, data in all_data.items():
            pick_count = len(data['actions']['pick'])
            place_count = len(data['actions']['place'])
            total_sequences = data['total_sequences']
            
            # Calculate total duration
            total_duration = sum(seq['duration_seconds'] for seq in data['file_info'])
            avg_duration = total_duration / total_sequences if total_sequences > 0 else 0
            
            # Collect all durations for this participant
            durations = [seq['duration_seconds'] for seq in data['file_info']]
            all_durations.extend(durations)
            
            action_counts['pick'] += pick_count
            action_counts['place'] += place_count
            
            participant_stats.append({
                'participant': participant,
                'pick': pick_count,
                'place': place_count,
                'total': total_sequences,
                'total_duration_min': total_duration / 60,
                'avg_duration_sec': avg_duration,
                'estimated_samples': data['total_samples'],
                'split': 'TEST' if participant.startswith('p7') else 'TRAIN'
            })
        
        # Create summary DataFrame
        stats_df = pd.DataFrame(participant_stats)
        print("\nParticipant Summary:")
        print(stats_df.to_string(index=False))
        
        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Total participants: {len(all_data)}")
        print(f"  Total sequences: {sum(s['total'] for s in participant_stats)}")
        print(f"  Pick actions: {action_counts['pick']}")
        print(f"  Place actions: {action_counts['place']}")
        print(f"  Average sequence duration: {np.mean(all_durations):.1f}s")
        print(f"  Total estimated samples: {sum(s['estimated_samples'] for s in participant_stats):,}")
        
        return stats_df, all_durations
    
    def create_visualizations(self, stats_df, all_durations):
        """Create visualizations of the dataset structure"""
        plt.figure(figsize=(15, 10))
        
        # Participant sequence counts
        plt.subplot(2, 3, 1)
        train_df = stats_df[stats_df['split'] == 'TRAIN']
        test_df = stats_df[stats_df['split'] == 'TEST']
        
        x_pos = range(len(stats_df))
        plt.bar([i for i, split in enumerate(stats_df['split']) if split == 'TRAIN'], 
                train_df['total'], label='TRAIN', alpha=0.7)
        plt.bar([i for i, split in enumerate(stats_df['split']) if split == 'TEST'], 
                test_df['total'], label='TEST', alpha=0.7)
        
        plt.xticks(x_pos, stats_df['participant'], rotation=45)
        plt.ylabel('Number of Sequences')
        plt.title('Sequences per Participant')
        plt.legend()
        
        # Action distribution
        plt.subplot(2, 3, 2)
        actions = ['pick', 'place']
        action_totals = [stats_df['pick'].sum(), stats_df['place'].sum()]
        plt.pie(action_totals, labels=actions, autopct='%1.1f%%')
        plt.title('Action Distribution')
        
        # Duration distribution
        plt.subplot(2, 3, 3)
        plt.hist(all_durations, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Count')
        plt.title('Sequence Duration Distribution')
        plt.axvline(np.mean(all_durations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_durations):.1f}s')
        plt.legend()
        
        # Estimated samples vs raw sequences
        plt.subplot(2, 3, 4)
        plt.scatter(stats_df['total'], stats_df['estimated_samples'], 
                   c=['red' if split == 'TEST' else 'blue' for split in stats_df['split']])
        plt.xlabel('Raw Sequences')
        plt.ylabel('Estimated Samples')
        plt.title('Raw Sequences vs Sliding Window Samples')
        
        # Add labels for test participants
        for i, row in stats_df.iterrows():
            if row['split'] == 'TEST':
                plt.annotate(row['participant'], 
                           (row['total'], row['estimated_samples']),
                           xytext=(5, 5), textcoords='offset points')
        
        # Train vs test comparison
        plt.subplot(2, 3, 5)
        train_total = train_df['total'].sum()
        test_total = test_df['total'].sum()
        train_samples = train_df['estimated_samples'].sum()
        test_samples = test_df['estimated_samples'].sum()
        
        categories = ['Raw Sequences', 'Estimated Samples']
        train_values = [train_total, train_samples]
        test_values = [test_total, test_samples]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, train_values, width, label='TRAIN')
        plt.bar(x + width/2, test_values, width, label='TEST')
        plt.xticks(x, categories)
        plt.ylabel('Count')
        plt.title('Train vs Test Comparison')
        plt.legend()
        
        # Sample inflation visualization
        plt.subplot(2, 3, 6)
        inflation_ratios = stats_df['estimated_samples'] / stats_df['total']
        plt.bar(range(len(stats_df)), inflation_ratios, 
                color=['red' if split == 'TEST' else 'blue' for split in stats_df['split']])
        plt.xticks(range(len(stats_df)), stats_df['participant'], rotation=45)
        plt.ylabel('Sample Inflation Ratio')
        plt.title('Sliding Window Sample Inflation')
        
        plt.tight_layout()
        plt.show()


def main():
    # Update this path to your GazeMotion preprocessed data
    data_dir = r"D:\GazeMotion\mogaze_processed"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please update the data_dir path")
        return
    
    analyzer = GazeMotionDatasetAnalyzer(data_dir)
    
    # Run analysis
    all_data, total_sequences, total_estimated_samples = analyzer.analyze_all_participants()
    
    if not all_data:
        print("No participant data found!")
        return
    
    # Compare sampling strategies
    test_sequences, test_samples_estimated = analyzer.compare_sampling_strategies(all_data)
    
    # Analyze sequence characteristics
    stats_df, all_durations = analyzer.analyze_sequence_characteristics(all_data)
    
    # Create visualizations
    analyzer.create_visualizations(stats_df, all_durations)
    
    # Final comparison summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Your Dataset (Preprocessed):")
    print(f"  Test sequences: 504")
    print(f"  One sequence per action segment")
    print(f"  Performance: 70.8mm MPJPE")
    
    print(f"\nGazeMotion Dataset (Preprocessed):")
    print(f"  Raw test sequences: {test_sequences}")
    print(f"  Sliding window samples: 22,901 (estimated: {test_samples_estimated:,})")
    print(f"  Performance: 84.8mm MPJPE")
    
    print(f"\nKey Insights:")
    print(f"  Data efficiency: Your method uses 45x fewer samples")
    print(f"  Performance advantage: 16.5% better accuracy")
    print(f"  Model efficiency: 9.3x fewer parameters (0.20M vs 1.86M)")
    print(f"  Training approach: No overlapping windows vs heavy overlap")


if __name__ == "__main__":
    main()