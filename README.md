# siMLPe with Gated Cross-Attention for Multimodal Human Motion Prediction

**Novel MLP-compatible gated cross-attention architecture for integrating gaze information into human motion prediction.**

## Overview

This repository extends the siMLPe architecture with novel gated cross-attention mechanisms to incorporate eye-gaze information for human motion prediction. Our approach achieves **70.8mm MPJPE** on the MoGaze dataset, representing a **16.5% improvement** over existing gaze-enhanced methods while using **9.3× fewer parameters**.

### Key Features

- **Novel Architecture**: First MLP-compatible gated cross-attention system
- **State-of-the-art Performance**: 70.8mm MPJPE on MoGaze dataset  
- **Efficient**: 0.20M parameters vs 1.86M in competing approaches
- **Robust Training**: Aggressive progressive training strategy preventing baseline degradation
- **Complete Implementation**: Includes baseline, training, evaluation, and analysis tools

## Results

| Method | MPJPE (mm) | Parameters | Improvement |
|--------|------------|------------|-------------|
| **Ours (siMLPe+Gaze)** | **70.8** | **0.20M** | **-16.5%** |
| GazeMotion | 84.8 | ~1.86M | baseline |
| siMLPe (pose-only) | 73.3 | 0.14M | -13.5% |

**Temporal Breakdown:**
- 200ms: 19.1mm (33.2% better than GazeMotion)
- 400ms: 46.1mm (23.0% better than GazeMotion) 
- 1000ms: 161.7mm (9.5% better than GazeMotion)

## Installation

### Requirements

```bash
# Core dependencies
torch >= 1.8.0
numpy >= 1.19.0
scipy >= 1.6.0
pandas >= 1.2.0
h5py >= 3.0.0
tqdm >= 4.60.0
tensorboard >= 2.5.0
einops >= 0.3.0
```

### Setup

```bash
# Clone repository
git clone https://github.com/dulucas/siMLPe.git
cd siMLPe

# Install dependencies
pip install torch torchvision numpy scipy pandas tqdm tensorboard h5py einops

# Create data directories
mkdir -p data/mogaze_3d
```

## Dataset Preparation

### MoGaze Dataset

1. Download the MoGaze dataset from the [official source](https://humans-to-robots-motion.github.io/mogaze)
2. Place raw HDF5 files in `data/mogaze/`
3. Run preprocessing (see `exps/mogaze_baseline_3d/mogaze_preprocessor.py`)

**Expected data structure:**
```
data/mogaze_3d/
├── p1_1_action1_pose_3d.npy    # 3D joint positions [frames, 63]
├── p1_1_action1_gaze.npy       # Gaze directions [frames, 3]
├── p1_1_action1_metadata.npy   # Sequence metadata
└── ...
```

## Quick Start

### 1. Baseline (Pose-only)

```bash
cd exps/mogaze_baseline_3d

# Train baseline model
python train_mogaze_3d.py --with-normalization --num 48

# Evaluate baseline
python test_mogaze_3d.py --model-pth log_3d/snapshot/model-iter-40000.pth
```

Expected result: ~73.3mm MPJPE

### 2. Breakthrough (Gaze-enhanced)

```bash
cd exps/mogaze_baseline_gaze

# Train with aggressive strategy
python train_mogaze_aggressive.py --seed 999 --gate-lr-multiplier 10.0

# Evaluate breakthrough model  
python test_mogaze_aggressive.py --model-pth log_aggressive/snapshot/model-final-aggressive.pth
```

Expected result: ~70.8mm MPJPE with gate weight ~0.359

## Architecture

### Gated Cross-Attention Framework

```
Input: 66D (63 pose + 3 gaze)
    ↓
┌─ Pose Pathway: siMLPe MLP ────────── Pose Features (63D)
│                                            ↓
└─ Gaze Pathway: GazeEncoder ── Gaze Features (64D)
                                            ↓
                              Cross-Attention Module
                                            ↓
                              Gated Fusion (α = 0.359)
                                            ↓
                              Output: 63D Poses
```

### Key Components

- **GazeEncoder**: Processes 3D gaze with temporal smoothing and velocity features
- **GatedCrossAttention**: Multi-head attention where gaze queries pose features
- **Progressive Gating**: Learnable gate starting at 0, trained with aggressive strategy

## Training Strategy

### Aggressive Progressive Training

1. **Baseline Stage** (5K iter): Gate frozen at 0, establish pose-only performance
2. **Aggressive Stage** (20K iter): High gate learning rates (×10), regularization  
3. **Refinement Stage** (15K iter): Full multimodal training with alignment losses

**Key Innovation**: Aggressive training achieves gate weight 0.359 vs 0.0001 in conservative approaches.

## Repository Structure

```
siMLPe/
├── lib/utils/                          # Shared utilities
├── exps/
│   ├── mogaze_baseline_3d/            # Pose-only baseline (73.3mm)
│   │   ├── train_mogaze_3d.py        # Baseline training
│   │   ├── test_mogaze_3d.py         # Baseline evaluation  
│   │   └── mogaze_config_3d.py       # Configuration
│   └── mogaze_baseline_gaze/          # Multimodal breakthrough (70.8mm)
│       ├── gated_attention_model.py  # Novel architecture
│       ├── aggressive_gate_trainer.py # Training strategy
│       ├── train_mogaze_aggressive.py # Training script
│       ├── test_mogaze_aggressive.py  # Evaluation script
│       └── mogaze_dataset_gaze.py     # Data loading
└── data/                              # Dataset storage
```

## Pre-trained Models

Download pre-trained models from [todo](todo):

- `model-baseline-3d.pth`: Pose-only baseline (73.3mm MPJPE)
- `model-final-aggressive.pth`: Breakthrough gaze-enhanced model (70.8mm MPJPE)

## Evaluation

### Standard Evaluation

```bash
python test_mogaze_aggressive.py --model-pth path/to/model.pth
```

### Comprehensive Analysis

```bash
python test_mogaze_aggressive.py \
    --model-pth path/to/model.pth \
    --analyze-attention \
    --save-results results.csv
```

Outputs:
- MPJPE metrics with confidence intervals
- Gate utilization analysis  
- Attention pattern visualization
- Efficiency benchmarks



### Performance Expectations

- **Training time**: ~2-3 hours for full 40K iterations on GTX 1650
- **Memory usage**: ~1.5GB GPU memory during training
- **Expected results**: 70.8±66.5mm MPJPE with 0.359 gate weight


## Acknowledgments

- Original siMLPe implementation by [Guo et al.](https://github.com/dulucas/siMLPe)
- MoGaze dataset by [Kratzer et al.](https://sites.google.com/view/mogaze)  
- Research conducted during internship at [DIIM Lab UofGuelph]

## Contact

For questions about the gated cross-attention extensions, please open an issue or contact [uasim.bese21seecs@seecs.edu.pk].

---

*This work extends the siMLPe architecture with novel multimodal capabilities, achieving state-of-the-art performance on human motion prediction with gaze information.*
