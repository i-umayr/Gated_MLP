# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research fork of siMLPe (MLP-based human motion prediction) adapted to the **MoGaze** dataset, investigating whether eye-gaze improves 3D pose forecasting. The main contribution is a **gated cross-attention** fusion of gaze into siMLPe, compared against a pose-only siMLPe baseline and fusion ablations (concat / additive / cross-attention without gate) under leave-one-participant-out cross-validation (LOPOCV).

`exps/` holds five experiments: `simlpe_baseline/` and `gated_gaze/` (one folder per fold) and the three fusion variants `mogaze_baseline_concat/`, `_additive/`, `_no_gate/`. Only `lib/utils/logger.py` and `lib/utils/pyt_utils.py` are still used (imported as `utils.*` via the configs' `add_path(root/lib)`).

There is no test suite, linter, or build step. "Testing" means evaluating trained checkpoints.

## Running things

Every experiment folder is self-contained: scripts use relative paths (`log_*/snapshot/...`, `runs/...`, the `--exp-name` .txt log), so **always `cd` into the experiment folder before running**. Training requires CUDA (`.cuda()` is hard-coded). Root-level scripts (`mogaze_preprocessor.py`, `flops.py`) run from the repo root.

Dependencies (pinned in README.md): `torch`, `numpy`, `scipy`, `easydict`, `einops`, `tqdm`, `tensorboard`, `pandas`, `h5py` (preprocessing), `fvcore` (`flops.py`).

```bash
# Pose-only siMLPe baseline (one fold; fold_<p> holds out participant <p>)
cd exps/simlpe_baseline/fold_p1
python train_mogaze_3d.py --with-normalization          # -> log_3d/snapshot/model-iter-*.pth
python test_mogaze_3d.py --model-pth log_3d/snapshot/model-iter-80000.pth

# Gated gaze model (one fold) - the "aggressive" trainer is the one used for reported results
cd exps/gated_gaze/fold_p1
python train_mogaze_aggressive.py --with-normalization  # -> log_aggressive/snapshot/model-final-aggressive.pth
python test_mogaze_aggressive.py --model-pth log_aggressive/snapshot/model-final-aggressive.pth

# Ablations: one script trains all 6 folds sequentially (or a subset)
cd exps/mogaze_baseline_concat          # same pattern in _additive and _no_gate
python train_mogaze_concat.py --folds p1 p2             # -> log_concat/snapshot/fold_<p>/model-final.pth
python test_mogaze_concat.py --folds p1 p2

# Paper results (run from the repo root): MPJPE of every model on all folds; stride 1 only for simlpe and gated
python eval_all.py                                      # -> eval_all_folds_results.csv, eval_all_folds_per_frame.csv (repo root)
python eval_all.py --models simlpe gated progressive concat additive crossattn --strides 10

# Parameters and FLOPs of siMLPe and the gated MLP (Table III; run from the repo root)
python flops.py
```

Check each script's argparse for the remaining flags. Most model hyperparameters live in the folder's `mogaze_config_*.py`, an `EasyDict` that scripts import as `config` and then override from the CLI (`--num`, `--with-normalization`, `--spatial-fc`, `--temporal-only`, `--layer-norm-axis`).

## Data

- Raw MoGaze HDF5 files live in `data/mogaze/`. `python mogaze_preprocessor.py` (repo root; `--input-dir` / `--output-dir` default to `data/mogaze` and `data/mogaze_3d`) downsamples 120 → 30 Hz, keeps action segments with good gaze confidence, converts Euler angles to 3D joint positions, and segments them into `data/mogaze_3d/<session>_<task>_<idx>_{pose_3d,gaze,pose_euler,metadata}.npy`.
- `*_pose_3d.npy` is `[frames, 63]`: 21 joints × xyz, in meters. `*_gaze.npy` is `[frames, 3]`, a gaze direction vector. Gaze models concatenate them to 66-D. Inputs are 66-D and predictions are always 63-D pose only.
- Participants/sessions: p1→{p1_1,p1_2}, p2→{p2_1}, p4→{p4_1}, p5→{p5_1}, p6→{p6_1,p6_2}, p7→{p7_1,p7_3}. p3 is excluded because of eye-tracker calibration problems.
- `data/`, `*.npy`, `*.log`, `runs/` and `exps/**/*.txt` are gitignored. `*.pth` is ignored too, except the 36 released final checkpoints, which are whitelisted by explicit `!` rules in `.gitignore`.

## Architecture: the parts that span files

**Shared prediction pipeline** (duplicated in every train/test script and in `eval_all.py`): the 50-frame input is DCT-transformed along time (`get_dct_matrix`), passed through the model, inverse-DCT'd, truncated to the target length, and the last observed pose frame is added as a residual offset (`deriv_input` / `deriv_output`). Training predicts 10 frames. Evaluation predicts 30 frames (1 s at 30 fps) **auto-regressively**, 10 frames per step; the gaze model holds the last observed gaze constant across rollout steps. MPJPE is reported in mm at 200/400/600/800/1000 ms (frame indices 5, 11, 17, 23, 29). The loss is the per-joint L2 position error plus a velocity term when `use_relative_loss` is set. Configs keep the H36M field names (`h36m_input_length`, `h36m_target_length_train/eval`) because the model code reads those names.

**Gated model** (`gated_attention_model.py`, `SiMLPeWithGatedGaze`): the input is split into pose[:63] and gaze[63:66]. Pose goes through the siMLPe MLP stack (`mlp.py`, 63-D hidden). Gaze goes through `GazeEncoder` (smoothing conv, velocity features, 64-D). `GatedCrossAttention` lets gaze query pose features, with spatial weights and an uncertainty net. A scalar learnable `gaze_gate` (sigmoid) blends pose-only and gaze-enhanced features. The optimizer keeps `gaze_gate` in its own param group so it can get a separate LR multiplier. `get_gate_weight()` is logged everywhere as the measure of how much gaze the model uses.

**Staged training**: `AggressiveGateTrainer` (`aggressive_gate_trainer.py`) and `ProgressiveGazeTrainer` (`progressive_trainer.py`) run the curriculum stages (baseline → gate opening with gate regularization and alignment loss → full), switch stages by iteration count, and compute the loss. `train_mogaze_aggressive.py` uses the first; `train_mogaze_gated.py --progressive` uses the second. Note that `mogaze_config_gated.py` sets `log_dir='log_gated'`, and the aggressive trainer rewrites that to `log_aggressive` by string replacement.

## LOPOCV layout (important)

There are two parallel styles of cross-validation:

1. **Copy-per-fold folders** (baseline and gated model): `exps/simlpe_baseline/fold_<p>/` and `exps/gated_gaze/fold_<p>/` (p ∈ p1, p2, p4, p5, p6, p7) are full copies of the experiment, each holding out participant `<p>`. They differ **only** in the `TRAIN_SESSIONS` / `TEST_SESSIONS` constants at the top of `mogaze_dataset_3d.py` / `mogaze_dataset_gaze.py`; every other file is byte-identical across the six folds, so edit `fold_p1` and copy the file to the other folds. The training recipe is cosine LR with 2k warmup, 80k iterations, and the aggressive trainer starting from scratch (`--pretrained-baseline` default `None`). **A change to training/model code meant for the paper must be applied to all six fold folders.** `eval_all.py` imports each model folder's own config, model, and checkpoint, and purges `sys.modules` between folds so the duplicated module names don't collide.
2. **Parameterized folds** (ablations `mogaze_baseline_concat`, `_additive`, `_no_gate`): one folder with a `PARTICIPANT_SESSIONS` / `ALL_PARTICIPANTS` mapping in the dataset module and a `--folds` CLI. Checkpoints go to `log_<variant>/snapshot/fold_<p>/`.

When comparing models, report the **sample-weighted** average across folds (the paper / GazeMotion convention). `eval_all.py` prints both the weighted and unweighted averages. Test-window stride 10 is the current paper protocol; stride 1 is dense, GazeMotion-style evaluation.

The per-fold test scripts in `simlpe_baseline/` and `gated_gaze/` average the per-horizon (200–1000 ms) errors over **batches** (the last, smaller batch counts as much as a full one); their overall MPJPE is per-sample. The fusion-variant test scripts and `eval_all.py` average everything per sample. **`eval_all.py` (repo root) is the single source of the paper's MPJPE numbers**; the paper values are being updated to match its output, so don't change its windowing, rollout or metric without the user asking.

The released code is meant to reproduce the paper's trained models exactly. When cleaning or refactoring, keep model attribute names (checkpoints load with `strict=True`), the order of RNG use (model init → dataset → DataLoader shuffle), and the training-schedule logic unchanged. Verify by re-evaluating the checkpoints and comparing against the previous output.
