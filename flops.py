"""
Parameters and FLOPs of the siMLPe baseline and the gated MLP.

FLOPs/pass is counted with fvcore (one multiply-accumulate = one FLOP). The task
cost adds the DCT/IDCT matrix products applied outside the network and covers the
three auto-regressive steps (10 frames each) needed for the 30-frame horizon.

Usage (from the repository root):
    python flops.py
    python flops.py --breakdown      # also print the fvcore per-module table
"""
import argparse
import importlib
import os
import sys

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET_LEN = 30  # frames (1 s at 30 fps)

# The architecture is the same in every fold, so any fold folder can be used
MODELS = [
    dict(label="siMLPe (no gaze)",
         folder=os.path.join(ROOT, "exps", "simlpe_baseline", "fold_p1"),
         config_module="mogaze_config_3d", model_module="model",
         build=lambda mod, cfg: mod.siMLPe(cfg),
         input_dim=63, output_dim=63),
    dict(label="Gated MLP",
         folder=os.path.join(ROOT, "exps", "gated_gaze", "fold_p1"),
         config_module="mogaze_config_gated", model_module="gated_attention_model",
         build=lambda mod, cfg: mod.SiMLPeWithGatedGaze(cfg),
         input_dim=66, output_dim=63),
]

# Module names shared by the experiment folders; cleared before each import
SHARED_MODULES = ["model", "mlp", "gated_attention_model", "mogaze_config_3d", "mogaze_config_gated"]


def load_model(spec):
    for name in SHARED_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, spec["folder"])
    try:
        config = importlib.import_module(spec["config_module"]).config
        model_mod = importlib.import_module(spec["model_module"])
    finally:
        sys.path.remove(spec["folder"])
    model = spec["build"](model_mod, config)
    model.eval()
    return model, config


def count(spec, breakdown=False):
    model, config = load_model(spec)
    T = config.motion.h36m_input_length            # 50 observed frames
    step = config.motion.h36m_target_length_train  # 10 frames per step
    num_iterations = TARGET_LEN // step            # 3 auto-regressive steps

    analysis = FlopCountAnalysis(model, (torch.randn(1, T, spec["input_dim"]),))
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    pass_flops = analysis.total()

    # DCT on the [T, input_dim] input and IDCT on the [T, output_dim] output
    transform_flops = T * T * spec["input_dim"] + T * T * spec["output_dim"]
    task_flops = (pass_flops + transform_flops) * num_iterations

    if breakdown:
        print(f"\n{spec['label']}")
        print(flop_count_table(analysis, max_depth=2))

    return dict(label=spec["label"],
                params=sum(p.numel() for p in model.parameters()),
                pass_flops=pass_flops, transform_flops=transform_flops,
                iterations=num_iterations, task_flops=task_flops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakdown", action="store_true", help="print the fvcore per-module table")
    args = parser.parse_args()

    rows = [count(spec, args.breakdown) for spec in MODELS]

    print(f"\n{'Method':<18}{'Params':>12}{'FLOPs/pass':>14}{'DCT/IDCT':>12}{'Steps':>7}{'Task FLOPs':>14}")
    print("-" * 77)
    for r in rows:
        print(f"{r['label']:<18}{r['params'] / 1e6:>11.2f}M{r['pass_flops'] / 1e6:>13.2f}M"
              f"{r['transform_flops'] / 1e6:>11.2f}M{r['iterations']:>7}{r['task_flops'] / 1e6:>13.2f}M")
    for r in rows:
        print(f"{r['label']}: {r['params']:,} parameters, {r['pass_flops']:,} FLOPs/pass, "
              f"{r['task_flops']:,} task FLOPs")


if __name__ == "__main__":
    main()
