"""Analysis / summary for the facial keypoint study."""

import json
import math
import os
from glob import glob

import pandas as pd

from ..paths import AGGREGATE_DIR as PAPER_DIR
from ..paths import RESULTS_DIR as RESULTS_ROOT

BLOCKS = ["01_naive", "02_dedup", "03_aug"]


def _safe_nan(x):
    """Convert JSON NaN sentinels (python nan / string 'NaN') to float('nan')."""
    if x is None:
        return float("nan")
    if isinstance(x, float) and math.isnan(x):
        return float("nan")
    if isinstance(x, str) and x.lower() == "nan":
        return float("nan")
    return x


def _read_epoch_log(path):
    """jsonl -> list of dict with NaN-safe cohort columns."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # NaN is not valid JSON; be lenient.
            line = line.replace(": NaN", ": null")
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _extract_cohort(cohort_val, key):
    """Extract per-cohort mse; cohort_val may be None or missing 'hard'."""
    if not cohort_val or not isinstance(cohort_val, dict):
        return float("nan")
    d = cohort_val.get(key)
    if not d:
        return float("nan")
    return _safe_nan(d.get("mse"))


def build_block(block_name):
    block_dir = os.path.join(RESULTS_ROOT, block_name)
    json_files = sorted(glob(os.path.join(block_dir, "results", "*.json")))
    if not json_files:
        print(f"[warn] no JSON in {block_dir}")
        return None, None

    summary_rows = []
    epoch_rows = []

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)

        key = d["key"]
        cell_core = {
            "dataset": d["dataset"],
            "model": d["model"],
            "strategy": d["strategy"],
            "seed": d["seed"],
        }

        # Read epoch log to fill best_epoch, final_train_mse, final_val_mse,
        # best_val_rmse_pix (matching best_val_mse epoch).
        log_path = os.path.join(block_dir, "logs", f"{key}.jsonl")
        epochs = _read_epoch_log(log_path)

        best_val_mse = _safe_nan(d.get("best_val_mse"))
        best_epoch = None
        best_val_rmse_pix = float("nan")
        final_train_mse = float("nan")
        final_val_mse = float("nan")

        if epochs:
            # find epoch with minimum val_mse
            best_rec = min(
                epochs,
                key=lambda r: (
                    float("inf") if r.get("val_mse") is None else r["val_mse"]
                ),
            )
            best_epoch = int(best_rec["epoch"])
            best_val_rmse_pix = _safe_nan(best_rec.get("val_rmse_pix"))
            final_train_mse = _safe_nan(epochs[-1].get("train_mse"))
            final_val_mse = _safe_nan(epochs[-1].get("val_mse"))

        summary_rows.append(
            {
                **cell_core,
                "n_train": d.get("n_train"),
                "n_val": d.get("n_val"),
                "use_mask": d.get("use_mask", False),
                "best_val_mse": best_val_mse,
                "best_epoch": best_epoch,
                "best_val_rmse_pix": best_val_rmse_pix,
                "final_val_mse": final_val_mse,
                "final_train_mse": final_train_mse,
                "tr_frac_frontal": d.get("tr_cohort_frac", {}).get("frontal"),
                "tr_frac_asym": d.get("tr_cohort_frac", {}).get("asym"),
                "tr_frac_hard": d.get("tr_cohort_frac", {}).get("hard"),
                "elapsed_sec": d.get("elapsed_sec"),
                "block": block_name,
            }
        )

        for r in epochs:
            cv = r.get("cohort_val")
            epoch_rows.append(
                {
                    **cell_core,
                    "epoch": int(r["epoch"]),
                    "train_mse": _safe_nan(r.get("train_mse")),
                    "val_mse": _safe_nan(r.get("val_mse")),
                    "val_rmse_pix": _safe_nan(r.get("val_rmse_pix")),
                    "cohort_frontal_mse": _extract_cohort(cv, "frontal"),
                    "cohort_asym_mse": _extract_cohort(cv, "asym"),
                    "cohort_hard_mse": _extract_cohort(cv, "hard"),
                    "sec": _safe_nan(r.get("sec")),
                    "block": block_name,
                }
            )

    summary = pd.DataFrame(summary_rows)
    epoch = pd.DataFrame(epoch_rows)

    tables_dir = os.path.join(block_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    summary_path = os.path.join(tables_dir, "summary_df.csv")
    epoch_path = os.path.join(tables_dir, "epoch_df.csv")
    summary.to_csv(summary_path, index=False)
    epoch.to_csv(epoch_path, index=False)
    print(f"[{block_name}] summary: {len(summary)} rows -> {summary_path}")
    print(f"[{block_name}] epoch  : {len(epoch)} rows -> {epoch_path}")
    return summary, epoch


def build_paper_bundle(all_summary, all_epoch):
    os.makedirs(PAPER_DIR, exist_ok=True)
    long_path = os.path.join(PAPER_DIR, "tier1_full_long.csv")
    all_summary.to_csv(long_path, index=False)
    print(f"[paper] long table -> {long_path} ({len(all_summary)} rows)")

    epoch_path = os.path.join(PAPER_DIR, "tier1_full_epoch.csv")
    all_epoch.to_csv(epoch_path, index=False)
    print(f"[paper] epoch table -> {epoch_path} ({len(all_epoch)} rows)")


def _main_build_summary():
    per_block = {}
    for blk in BLOCKS:
        s, e = build_block(blk)
        if s is not None:
            per_block[blk] = (s, e)

    if not per_block:
        print("[error] no blocks found")
        return

    all_summary = pd.concat([v[0] for v in per_block.values()], ignore_index=True)
    all_epoch = pd.concat([v[1] for v in per_block.values()], ignore_index=True)

    build_paper_bundle(all_summary, all_epoch)

    # sanity: expected 60 cells (9+6+45)
    expected = {"01_naive": 9, "02_dedup": 6, "03_aug": 45}
    for blk, n_expected in expected.items():
        n = (all_summary["block"] == blk).sum() if blk in per_block else 0
        flag = "OK" if n == n_expected else "MISMATCH"
        print(f"  [{flag}] {blk}: {n}/{n_expected} cells")
