
<<<<<<< HEAD
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
>>>>>>> ebf94df9fd3add9ebdbd505c15462d90e7f4b806
