"""
eval_all.py
===========

Reproduces the MPJPE results of the paper from the released checkpoints.

Evaluates every model of the paper on all six LOPOCV folds with a configurable
test-window stride, using ONE shared windowing + metric code path so the models
are directly comparable:

    simlpe       siMLPe baseline (no gaze)
    gated        gated MLP, aggressive gate-opening schedule
    progressive  gated MLP, progressive gate-opening schedule
    concat       fusion variant: input-level concatenation
    additive     fusion variant: additive fusion
    crossattn    fusion variant: cross-attention without gate

Paper results produced by this script:
    * stride 10, all models: per-fold and weighted MPJPE (Tables I, II and IV,
      and the per-fold values used in the paired t-test)
    * stride 1, siMLPe and the aggressive gated MLP only ("Sensitivity to the
      Evaluation Protocol")

Expected layout (relative to this file, the repository root):

    eval_all.py                               <- this file
    exps/
        simlpe_baseline/fold_p1 ... fold_p7   (siMLPe baseline)
        gated_gaze/fold_p1 ... fold_p7        (gated model: log_aggressive/ and log_gated/)
        mogaze_baseline_concat/               (log_concat/snapshot/fold_<p>/)
        mogaze_baseline_additive/             (log_additive/snapshot/fold_<p>/)
        mogaze_baseline_no_gate/              (log_crossattn/snapshot/fold_<p>/)

    Fold <p> holds out participant <p> for testing.

Usage (from the repository root):
    python eval_all.py                                   # all models (stride 1 only for simlpe, gated)
    python eval_all.py --strides 10                      # paper protocol only
    python eval_all.py --models simlpe gated progressive # Tables I and II
    python eval_all.py --models concat additive crossattn
    python eval_all.py --device cpu

Per fold, for each model:
  * imports that folder's own config / model code / checkpoint,
  * builds the held-out participant's test windows exactly like the original
    eval datasets (MoGaze3DEval / MoGazeGazeEval: same files, 50+30 = 80-frame
    windows), but with the stride given on the command line
    (10 = current paper protocol, 1 = dense, GazeMotion-style),
  * runs the same auto-regressive rollout as the original test scripts
    (10 frames per step; for the gated model the last observed gaze is held
    constant),
  * accumulates errors PER SAMPLE, so the horizon columns and the overall
    average are computed over exactly the same samples.

Outputs: printed tables + CSV files in the repository root (change with --out-dir)
    eval_all_folds_results.csv     (per fold x model x stride, plus averages)
    eval_all_folds_per_frame.csv   (all 30 future frames)
"""

import argparse
import glob
import importlib
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Folds and model specifications
# --------------------------------------------------------------------------
FOLDS = [  # (fold folder, fold name, test recordings)
    ("fold_p1", "p1", ["p1_1", "p1_2"]),
    ("fold_p2", "p2", ["p2_1"]),
    ("fold_p4", "p4", ["p4_1"]),
    ("fold_p5", "p5", ["p5_1"]),
    ("fold_p6", "p6", ["p6_1", "p6_2"]),
    ("fold_p7", "p7", ["p7_1", "p7_3"]),
]

# per_fold_code=True:  <model_dir>/fold_<p>/ holds the code and the checkpoint of each fold
# per_fold_code=False: <model_dir>/ holds the code; ckpt_rel contains a {fold} placeholder
# strides: test-window strides evaluated for the model (stride 1 only for siMLPe and the aggressive gated MLP)
SPECS = {
    "simlpe": dict(
        label="siMLPe (no gaze)",
        per_fold_code=True,
        model_dir="simlpe_baseline",
        config_module="mogaze_config_3d",
        model_module="model",
        ckpt_rel=os.path.join("log_3d", "snapshot", "model-iter-80000.pth"),
        uses_gaze=False,
        strides=[10, 1],
        build=lambda mod, cfg: mod.siMLPe(cfg),
    ),
    "gated": dict(
        label="Gated MLP (Agg.)",
        per_fold_code=True,
        model_dir="gated_gaze",
        config_module="mogaze_config_gated",
        model_module="gated_attention_model",
        ckpt_rel=os.path.join("log_aggressive", "snapshot", "model-final-aggressive.pth"),
        uses_gaze=True,
        strides=[10, 1],
        build=lambda mod, cfg: mod.create_gated_model(cfg),
    ),
    "progressive": dict(
        label="Gated MLP (Prog.)",
        per_fold_code=True,
        model_dir="gated_gaze",
        config_module="mogaze_config_gated",
        model_module="gated_attention_model",
        ckpt_rel=os.path.join("log_gated", "snapshot", "model-iter-80000.pth"),
        uses_gaze=True,
        strides=[10],
        build=lambda mod, cfg: mod.create_gated_model(cfg),
    ),
    "concat": dict(
        label="Concatenation",
        per_fold_code=False,
        model_dir="mogaze_baseline_concat",
        config_module="mogaze_config_concat",
        model_module="concat_model",
        ckpt_rel=os.path.join("log_concat", "snapshot", "fold_{fold}", "model-final.pth"),
        uses_gaze=True,
        strides=[10],
        build=lambda mod, cfg: mod.SiMLPeConcatFusion(cfg),
    ),
    "additive": dict(
        label="Additive fusion",
        per_fold_code=False,
        model_dir="mogaze_baseline_additive",
        config_module="mogaze_config_additive",
        model_module="additive_model",
        ckpt_rel=os.path.join("log_additive", "snapshot", "fold_{fold}", "model-final.pth"),
        uses_gaze=True,
        strides=[10],
        build=lambda mod, cfg: mod.SiMLPeAdditiveFusion(cfg),
    ),
    "crossattn": dict(
        label="Cross-attn. (no gate)",
        per_fold_code=False,
        model_dir="mogaze_baseline_no_gate",
        config_module="mogaze_config_crossattn",
        model_module="crossattn_model",
        ckpt_rel=os.path.join("log_crossattn", "snapshot", "fold_{fold}", "model-final.pth"),
        uses_gaze=True,
        strides=[10],
        build=lambda mod, cfg: mod.SiMLPeCrossAttnNoGate(cfg),
    ),
}

# Module names shared by the experiment folders; purged between folds/models
FOLD_MODULES = ["mogaze_config_3d", "mogaze_config_gated", "mogaze_config_concat",
                "mogaze_config_additive", "mogaze_config_crossattn",
                "model", "gated_attention_model", "concat_model", "additive_model",
                "crossattn_model", "mlp",
                "mogaze_dataset_3d", "mogaze_dataset_gaze", "mogaze_dataset_concat",
                "mogaze_dataset_additive", "mogaze_dataset_crossattn"]

KEY_FRAMES = [5, 11, 17, 23, 29]            # 0-indexed -> 200..1000 ms at 30 fps
KEY_TIMES = [200, 400, 600, 800, 1000]
COLS = [f"{t}ms" for t in KEY_TIMES] + ["mpjpe"]
N_FUTURE = 30


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_dct_matrix(N):
    """Identical to get_dct_matrix in the original test scripts."""
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def import_fold_modules(fold_dir, config_module, model_module):
    for name in FOLD_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, fold_dir)
    try:
        cfg_mod = importlib.import_module(config_module)
        model_mod = importlib.import_module(model_module)
    finally:
        if sys.path and sys.path[0] == fold_dir:
            sys.path.pop(0)
    return cfg_mod.config, model_mod


def load_checkpoint(model, ckpt_path):
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:            # older PyTorch without the weights_only argument
        state = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def build_test_windows(anno_dir, test_recordings, input_len, stride, uses_gaze):
    """
    Same file selection and windowing as MoGaze3DEval (uses_gaze=False) or
    MoGazeGazeEval (uses_gaze=True), with a configurable stride.
    """
    all_pose_files = sorted(glob.glob(os.path.join(anno_dir, "*_pose_3d.npy")))
    files = []
    for rec in test_recordings:
        for pose_file in all_pose_files:
            if os.path.basename(pose_file).startswith(rec):
                if uses_gaze:
                    gaze_file = pose_file.replace("_pose_3d.npy", "_gaze.npy")
                    if os.path.exists(gaze_file):
                        files.append((pose_file, gaze_file))
                else:
                    files.append((pose_file, None))

    seqs, index = [], []
    min_len = input_len + N_FUTURE
    n_short = 0
    for pose_file, gaze_file in files:
        data = np.load(pose_file)
        if uses_gaze:
            gaze = np.load(gaze_file)
            if data.shape[0] != gaze.shape[0]:
                continue
            data = np.concatenate([data, gaze], axis=1)      # [frames, 66]
        N = data.shape[0]
        if N < min_len:
            n_short += 1
            continue
        starts = np.arange(0, N - min_len + 1, stride)
        index.extend((len(seqs), int(s)) for s in starts)
        seqs.append(data.astype(np.float32))
    return seqs, index, len(files), n_short


@torch.no_grad()
def evaluate(model, config, seqs, index, device, batch_size, uses_gaze):
    """Auto-regressive rollout; returns per-sample per-frame errors [S, 30] in mm."""
    input_len = config.motion.h36m_input_length
    step = config.motion.h36m_target_length_train            # 10 frames per step
    num_step = 1 if step == N_FUTURE else N_FUTURE // step + 1   # as in original scripts

    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    dct_m = torch.tensor(dct_m).float().to(device).unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().to(device).unsqueeze(0)

    errs = []
    for b0 in range(0, len(index), batch_size):
        batch = index[b0:b0 + batch_size]
        win = np.stack([seqs[i][s:s + input_len + N_FUTURE] for i, s in batch])
        win = torch.from_numpy(win).to(device)
        motion_input = win[:, :input_len]                    # [B, 50, 63 or 66]
        target_pose = win[:, input_len:, :63]                # [B, 30, 63]

        current_input = motion_input.clone()
        outputs = []
        for _ in range(num_step):
            if config.deriv_input:
                x = torch.matmul(dct_m[:, :, :input_len], current_input.clone())
            else:
                x = current_input.clone()
            out = model(x)
            out = torch.matmul(idct_m[:, :input_len, :], out)[:, :step, :]
            if config.deriv_output:
                out = out + current_input[:, -1:, :63]
            outputs.append(out)
            if uses_gaze:
                last_gaze = current_input[:, -1:, 63:66].repeat(1, out.shape[1], 1)
                nxt = torch.cat([out, last_gaze], dim=2)
            else:
                nxt = out
            current_input = torch.cat([current_input[:, step:], nxt], dim=1)

        pred = torch.cat(outputs, dim=1)[:, :N_FUTURE]       # [B, 30, 63]
        B = pred.shape[0]
        e = torch.linalg.norm(pred.reshape(B, N_FUTURE, 21, 3)
                              - target_pose.reshape(B, N_FUTURE, 21, 3), dim=-1)
        errs.append((e.mean(dim=2) * 1000.0).cpu().numpy())  # [B, 30] in mm
    return np.concatenate(errs, axis=0)


def weighted(rows):
    n = np.array([r["samples"] for r in rows], float)
    M = np.array([[r[c] for c in COLS] for r in rows])
    return (M * n[:, None]).sum(0) / n.sum(), M.mean(0), int(n.sum())


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r.get(k, ''):.4f}" if isinstance(r.get(k, ""), float)
                             else str(r.get(k, "")) for k in keys) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "exps"),
                    help="folder containing the experiment folders")
    ap.add_argument("--out-dir", default=REPO_ROOT, help="folder for the CSV outputs")
    ap.add_argument("--models", nargs="+", default=list(SPECS.keys()),
                    choices=list(SPECS.keys()))
    ap.add_argument("--strides", type=int, nargs="+", default=[10, 1])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--simlpe-ckpt", default=None, help="override siMLPe checkpoint path (relative to fold folder)")
    ap.add_argument("--gated-ckpt", default=None, help="override gated checkpoint path (relative to fold folder)")
    ap.add_argument("--progressive-ckpt", default=None, help="override progressive checkpoint path (relative to fold folder)")
    ap.add_argument("--out-prefix", default="eval_all_folds")
    args = ap.parse_args()
    if args.simlpe_ckpt:
        SPECS["simlpe"]["ckpt_rel"] = args.simlpe_ckpt
    if args.gated_ckpt:
        SPECS["gated"]["ckpt_rel"] = args.gated_ckpt
    if args.progressive_ckpt:
        SPECS["progressive"]["ckpt_rel"] = args.progressive_ckpt

    device = torch.device(args.device)
    print("=" * 100)
    print("LOPOCV evaluation with configurable test stride")
    print(f"Root: {args.root}")
    print(f"Models: {args.models} | strides: {args.strides} | device: {device} | batch: {args.batch_size}")
    print("=" * 100)

    model_strides = {m: [s for s in args.strides if s in SPECS[m]["strides"]] for m in args.models}
    results = {m: {s: [] for s in model_strides[m]} for m in args.models}
    per_frame_rows = []

    for m in args.models:
        spec = SPECS[m]
        if not model_strides[m]:
            continue
        print(f"\n######## {spec['label']} ########")
        for fold_folder, fold, recs in FOLDS:
            if spec["per_fold_code"]:
                code_dir = os.path.join(args.root, spec["model_dir"], fold_folder)
            else:
                code_dir = os.path.join(args.root, spec["model_dir"])
            ckpt = os.path.join(code_dir, spec["ckpt_rel"].format(fold=fold))
            print(f"\n--- {m} | fold {fold} | test {recs}")
            if not os.path.isdir(code_dir):
                sys.exit(f"Folder not found: {code_dir}")
            if not os.path.exists(ckpt):
                sys.exit(f"Checkpoint not found: {ckpt}")

            config, model_mod = import_fold_modules(code_dir, spec["config_module"],
                                                     spec["model_module"])
            config.motion.h36m_target_length = config.motion.h36m_target_length_eval
            model = spec["build"](model_mod, config)
            load_checkpoint(model, ckpt)
            model.to(device).eval()
            gate = model.get_gate_weight() if hasattr(model, "get_gate_weight") else float("nan")
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    checkpoint: {ckpt}")
            print(f"    params: {n_params:,} | gate: {gate:.4f} | input length: "
                  f"{config.motion.h36m_input_length} | step: {config.motion.h36m_target_length_train}")

            for stride in model_strides[m]:
                seqs, index, n_files, n_short = build_test_windows(
                    config.mogaze_anno_dir, recs, config.motion.h36m_input_length,
                    stride, spec["uses_gaze"])
                if not index:
                    sys.exit(f"No test windows for {m} fold {fold} stride {stride}")
                t0 = time.time()
                err = evaluate(model, config, seqs, index, device, args.batch_size,
                               spec["uses_gaze"])
                fm = err.mean(axis=0)
                row = {"model": m, "fold": fold, "stride": stride, "files": n_files,
                       "too_short": n_short, "sequences": len(seqs),
                       "samples": err.shape[0], "gate": gate, "mpjpe": float(err.mean())}
                for f, t in zip(KEY_FRAMES, KEY_TIMES):
                    row[f"{t}ms"] = float(fm[f])
                results[m][stride].append(row)
                per_frame_rows.append({"model": m, "fold": fold, "stride": stride,
                                       "samples": err.shape[0],
                                       **{f"f{j+1}": float(fm[j]) for j in range(N_FUTURE)}})
                print(f"    stride {stride:>2}: files {n_files}, too short {n_short}, "
                      f"sequences {len(seqs)}, samples {err.shape[0]:>6} | "
                      f"MPJPE {row['mpjpe']:6.2f} | "
                      + "  ".join(f"{t}ms {row[f'{t}ms']:6.1f}" for t in KEY_TIMES)
                      + f" | {time.time() - t0:.1f}s")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ tables
    summary_rows = []
    for m in args.models:
        for stride in model_strides[m]:
            rows = results[m][stride]
            w, u, ntot = weighted(rows)
            tag = {10: "current paper protocol", 1: "dense, GazeMotion-style"}.get(stride, "custom")
            print("\n" + "=" * 100)
            print(f"{SPECS[m]['label']} - stride {stride} ({tag})")
            print("=" * 100)
            print(f"{'Fold':<6}{'Samples':>9}" + "".join(f"{c:>9}" for c in
                  ["200ms", "400ms", "600ms", "800ms", "1000ms", "Avg"]) + f"{'gate':>9}")
            print("-" * 100)
            for r in rows:
                print(f"{r['fold']:<6}{r['samples']:>9}" + "".join(f"{r[c]:>9.2f}" for c in COLS)
                      + f"{r['gate']:>9.4f}")
            print("-" * 100)
            print(f"{'Wtd':<6}{ntot:>9}" + "".join(f"{v:>9.2f}" for v in w)
                  + "   <- sample-weighted (paper / GazeMotion convention)")
            print(f"{'Mean':<6}{'':>9}" + "".join(f"{v:>9.2f}" for v in u)
                  + "   <- unweighted mean of the 6 folds")
            summary_rows += [{k: r[k] for k in ["model", "fold", "stride", "files", "too_short",
                                                "sequences", "samples", "gate"] + COLS}
                             for r in rows]
            summary_rows.append({"model": m, "fold": "weighted", "stride": stride,
                                 "samples": ntot, **dict(zip(COLS, map(float, w)))})
            summary_rows.append({"model": m, "fold": "unweighted", "stride": stride,
                                 "samples": ntot, **dict(zip(COLS, map(float, u)))})

    out1 = os.path.join(args.out_dir, f"{args.out_prefix}_results.csv")
    out2 = os.path.join(args.out_dir, f"{args.out_prefix}_per_frame.csv")
    write_csv(out1, summary_rows)
    write_csv(out2, per_frame_rows)
    print(f"\nSaved: {out1}\nSaved: {out2}")


if __name__ == "__main__":
    main()
