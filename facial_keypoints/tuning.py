"""Tuning for the facial keypoint study."""

import math
import os

import numpy as np

from .training import EPOCHS, train_one_run


def sample_param(spec, rng: np.random.Generator):
    kind = spec[0]
    if kind == "uniform":
        _, lo, hi = spec
        return float(rng.uniform(lo, hi))
    if kind == "loguniform":
        _, lo, hi = spec
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    if kind == "int":
        _, lo, hi = spec
        return int(rng.integers(lo, hi + 1))
    if kind == "choice":
        _, choices = spec
        return choices[int(rng.integers(0, len(choices)))]
    if kind == "fixed":
        return spec[1]
    raise ValueError(f"Unknown sampler kind: {kind!r}")


N_TRIALS_MLP = 8

N_TRIALS_CNN = 8

N_TRIALS_RESNET = 8

SPACE_MLP2 = {
    "lr": ("loguniform", 1e-4, 5e-3),
    "batch_size": ("choice", [64, 128, 256]),
    "weight_decay": ("choice", [0.0, 1e-5, 1e-4]),
    "model_hidden": ("choice", [256, 512, 1024]),
}

SPACE_CNN = {
    "lr": ("loguniform", 1e-4, 5e-3),
    "batch_size": ("choice", [64, 128, 256]),
    "weight_decay": ("choice", [0.0, 1e-5, 1e-4]),
}

SPACE_RESNET = {
    "lr": ("loguniform", 1e-5, 5e-4),
    "batch_size": ("choice", [64, 128]),
    "weight_decay": ("choice", [0.0, 1e-5, 1e-4]),
}

COMMON_BASE = {"epochs": 30, "patience": 8, "num_workers": 4}

SEARCH_SPACES = {
    "mlp2": SPACE_MLP2,
    "cnn_shallow": SPACE_CNN,
    "resnet18_pretrained": SPACE_RESNET,
}


def random_search(
    model: str,
    dataset: str,
    space: dict,
    n_trials: int,
    seed: int = 0,
    strategy: str = "none",
    out_dir: str = "./runs_hpsearch",
    epochs: int = None,
) -> list:
    """Random search over `space` for one (dataset, model) pair.

    Each trial samples one point from `space`, then calls `train_one_run()`
    with a per-trial out_dir. Returns trials sorted ascending by val_mse.
    Writes `search_results.csv` into `out_dir` summarising all trials.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    trials = []
    for t in range(n_trials):
        params = {k: sample_param(spec, rng) for k, spec in space.items()}
        run_id = f"trial_{t:03d}"
        cfg = {
            "dataset": dataset,
            "model": model,
            "strategy": strategy,
            "seed": int(params.get("seed", seed)),
            "out_dir": os.path.join(out_dir, run_id),
            "epochs": int(
                params.get("epochs", epochs if epochs is not None else EPOCHS)
            ),
            "overwrite": True,
            "hparams": params,
        }
        print(f"[hpsearch] trial {t + 1}/{n_trials}  params={params}")
        try:
            manifest = train_one_run(cfg)
        except Exception as e:
            print(f"[hpsearch] trial {t} FAILED: {e}")
            continue
        trials.append(
            {
                "trial": t,
                "run_id": run_id,
                "params": params,
                "val_mse": manifest.get("best_val_mse"),
                "test_mse": manifest.get("test_mse"),
            }
        )
    trials.sort(key=lambda r: (r["val_mse"] is None, r["val_mse"]))
    import csv as _csv

    summary_path = os.path.join(out_dir, "search_results.csv")
    fieldnames = ["trial", "run_id", "val_mse", "test_mse"] + sorted(space.keys())
    with open(summary_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in trials:
            row = {
                "trial": r["trial"],
                "run_id": r["run_id"],
                "val_mse": r["val_mse"],
                "test_mse": r["test_mse"],
            }
            row.update(r["params"])
            w.writerow(row)
    best = trials[0]["val_mse"] if trials else None
    print(f"[hpsearch] wrote {summary_path}  (best val_mse={best})")
    return trials
