# Gated MLP for Gaze-Pose Human Motion Prediction

Code for _Efficient Multimodal Human Motion Prediction: A Gated MLP Approach for Gaze-Pose Integration_. All models are trained and evaluated on MoGaze with leave-one-person-out cross-validation (LOPOCV).

## Requirements

Python 3.8 and a CUDA GPU (training scripts run on CUDA).

```bash
pip install torch==2.4.1 numpy==1.24.3 scipy==1.10.1 easydict==1.13 einops==0.8.1 h5py==3.11.0 \
            tqdm==4.67.1 tensorboard==2.14.0 pandas==2.0.3 fvcore==0.1.5.post20221221
```

## Data

1. Download MoGaze from https://humans-to-robots-motion.github.io/mogaze/ and place the HDF5 files (`p1_1_human_data.hdf5`, `p1_1_gaze_data.hdf5`, `p1_1_object_data.hdf5`, `p1_1_segmentations.hdf5`, …) in `data/mogaze/`.
2. Run the preprocessing from the repository root:

   ```bash
   python mogaze_preprocessor.py        # options: --input-dir data/mogaze --output-dir data/mogaze_3d
   ```

   This downsamples to 30 fps, keeps action segments with reliable gaze, converts joint angles to 3D positions and writes `data/mogaze_3d/<session>_<action>_<idx>_{pose_3d,gaze,pose_euler,metadata}.npy` (pose: 21 joints × 3; gaze: 3D direction).

## Reproducing the paper results

The final checkpoints of every model and fold are included in the repository, so all results can be reproduced without training. Run from the repository root:

```bash
python eval_all.py      # MPJPE of all models on all six folds
python flops.py         # parameters and FLOPs
```

| Paper result                                                     | Command                           |
| ---------------------------------------------------------------- | --------------------------------- |
| Tables I, II and IV (MPJPE, test window every 10 frames)         | `python eval_all.py --strides 10` |
| Sensitivity to the evaluation protocol (test window every frame) | `python eval_all.py --strides 1`  |
| Table III (parameters, FLOPs)                                    | `python flops.py`                 |

`eval_all.py` evaluates every model with one shared evaluation code path and prints, for each model, the per-fold MPJPE and the sample-weighted average over the six folds at 200, 400, 600, 800 and 1000 ms and over all 30 predicted frames. Stride 1 is evaluated for the siMLPe baseline and the aggressive gated MLP. The results are also saved to `eval_all_folds_results.csv` and `eval_all_folds_per_frame.csv`.

```bash
python eval_all.py --models simlpe gated progressive       # Tables I and II
python eval_all.py --models simlpe concat additive crossattn gated --strides 10   # Table IV
python eval_all.py --device cpu                            # without a GPU
```

Models: `simlpe` (siMLPe, no gaze), `gated` (gated MLP, aggressive schedule), `progressive` (gated MLP, progressive schedule), `concat`, `additive`, `crossattn` (fusion variants). All MPJPE values are in millimetres.

## Repository structure

```
.
├── eval_all.py                     MPJPE of every model on all folds (paper results)
├── flops.py                        parameters and FLOPs of siMLPe and the gated MLP
├── mogaze_preprocessor.py          raw MoGaze → data/mogaze_3d
├── data/
│   ├── mogaze/                     raw MoGaze HDF5 files (download)
│   └── mogaze_3d/                  preprocessed data (generated)
├── lib/utils/                      logging helpers used by the training scripts
└── exps/
    ├── simlpe_baseline/            siMLPe, pose only (no gaze)
    │   └── fold_p1 … fold_p7/      one self-contained folder per fold
    ├── gated_gaze/                 gated MLP (gaze + pose), aggressive and progressive training
    │   └── fold_p1 … fold_p7/      one self-contained folder per fold
    ├── mogaze_baseline_concat/     fusion variant: input-level concatenation
    ├── mogaze_baseline_additive/   fusion variant: additive fusion
    └── mogaze_baseline_no_gate/    fusion variant: cross-attention without gate
```

### Cross-validation folds

There are six folds, each named after the participant held out for testing (participant 3 is excluded):

| Fold | Test sessions | Fold | Test sessions |
| ---- | ------------- | ---- | ------------- |
| p1   | p1_1, p1_2    | p5   | p5_1          |
| p2   | p2_1          | p6   | p6_1, p6_2    |
| p4   | p4_1          | p7   | p7_1, p7_3    |

- `simlpe_baseline/` and `gated_gaze/` have one folder per fold (`fold_<p>`). The folders are identical except for the train/test split.
- The three fusion-variant folders train and evaluate all six folds from a single script; pass `--folds p1 p2 …` to run a subset.

## Training

Run every training script from inside its own folder. Logs, checkpoints and TensorBoard files (`runs/`) are written relative to that folder, at the paths `eval_all.py` loads from.

### siMLPe baseline (no gaze)

For each fold folder:

```bash
cd exps/simlpe_baseline/fold_p1
python train_mogaze_3d.py --with-normalization      # 80k iterations -> log_3d/snapshot/model-iter-80000.pth
```

### Gated MLP

For each fold folder:

```bash
cd exps/gated_gaze/fold_p1

# Aggressive gate-opening schedule (main model) -> log_aggressive/snapshot/model-final-aggressive.pth
python train_mogaze_aggressive.py --with-normalization

# Progressive gate-opening schedule -> log_gated/snapshot/model-iter-80000.pth
python train_mogaze_gated.py --progressive --with-normalization
```

### Fusion variants

| Variant                   | Folder                          | Train script                | Checkpoints                                       |
| ------------------------- | ------------------------------- | --------------------------- | ------------------------------------------------- |
| Concatenation             | `exps/mogaze_baseline_concat`   | `train_mogaze_concat.py`    | `log_concat/snapshot/fold_<p>/model-final.pth`    |
| Additive fusion           | `exps/mogaze_baseline_additive` | `train_mogaze_additive.py`  | `log_additive/snapshot/fold_<p>/model-final.pth`  |
| Cross-attention (no gate) | `exps/mogaze_baseline_no_gate`  | `train_mogaze_crossattn.py` | `log_crossattn/snapshot/fold_<p>/model-final.pth` |

```bash
cd exps/mogaze_baseline_concat
python train_mogaze_concat.py                       # all 6 folds, 80k iterations each
```

### Per-model test scripts

Each model folder also has a test script for a quick check of one fold (`test_mogaze_3d.py`, `test_mogaze_aggressive.py`, `test_mogaze_gated.py`) or of one fusion variant (`test_mogaze_concat.py`, `test_mogaze_additive.py`, `test_mogaze_crossattn.py`). Their overall MPJPE matches `eval_all.py`; the per-fold test scripts in `simlpe_baseline/` and `gated_gaze/` average the 200–1000 ms errors over batches instead of samples, so those columns can differ slightly. The paper results are the ones printed by `eval_all.py`.
=======
# siMLPe with Gated Cross-Attention for Multimodal Human Motion Prediction

**Novel MLP-compatible gated cross-attention architecture for integrating gaze information into human motion prediction.**

## Overview

This repository extends the siMLPe architecture with novel gated cross-attention mechanisms to incorporate eye-gaze information for human motion prediction. Our approach achieves **70.8mm MPJPE** on the MoGaze dataset, representing a **16.5% improvement** over existing gaze-enhanced methods while using **9.3× fewer parameters**.

### Key Features

- **Novel Architecture**: Novel MLP-compatible gated cross-attention system
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
>>>>>>> 6714c26c87fdf3c956e9d4100b7378bd9a755cb6
