"""Data prep / large for the facial keypoint study."""

import hashlib
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from ..paths import CLEAN_DIR, DATA_DIR

SRC = {
    "train": os.path.join(DATA_DIR, "data_large_train.csv"),
    "val": os.path.join(DATA_DIR, "data_large_val.csv"),
    "test": os.path.join(DATA_DIR, "data_large_test.csv"),
}

OUT_TEMPLATE = os.path.join(CLEAN_DIR, "data_large_clean_dedup_{split}.csv")

REPORT_PATH = os.path.join(CLEAN_DIR, "data_large_dedup_report.json")

KEYPOINT_COLS_LARGE = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
]

assert len(KEYPOINT_COLS_LARGE) == 8


def sha1_image_column(series: pd.Series) -> pd.Series:
    """SHA1 fingerprint of the raw pixel string (96*96 = 9216 ints, space-separated)."""
    return series.astype(str).map(lambda s: hashlib.sha1(s.encode()).hexdigest())


def _load_large_splits() -> dict:
    splits = {}
    for k, p in SRC.items():
        df = pd.read_csv(p)
        assert "Image" in df.columns, f"{p} missing Image column"
        for c in KEYPOINT_COLS_LARGE:
            assert c in df.columns, f"{p} missing {c}"
        splits[k] = df
    return splits


def dedup_alpha(splits: dict) -> tuple:
    """Protect test > val > train. Attach __sha1__ internally, return clean dict + drop counts."""
    for k, df in splits.items():
        df["__sha1__"] = sha1_image_column(df["Image"])

    test_sha = set(splits["test"]["__sha1__"])
    val_sha = set(splits["val"]["__sha1__"])

    tr = splits["train"]
    tr_mask = ~tr["__sha1__"].isin(test_sha | val_sha)
    tr_drop = int((~tr_mask).sum())
    tr_clean = tr.loc[tr_mask].reset_index(drop=True)

    va = splits["val"]
    va_mask = ~va["__sha1__"].isin(test_sha)
    va_drop = int((~va_mask).sum())
    va_clean = va.loc[va_mask].reset_index(drop=True)

    te_clean = splits["test"].reset_index(drop=True)

    return {
        "train": tr_clean,
        "val": va_clean,
        "test": te_clean,
    }, {"train_dropped": tr_drop, "val_dropped": va_drop, "test_dropped": 0}


def verify_no_residual_overlap(splits: dict) -> dict:
    """Post-dedup invariant: zero cross-split SHA1 overlap."""
    tr_sha = set(splits["train"]["__sha1__"])
    va_sha = set(splits["val"]["__sha1__"])
    te_sha = set(splits["test"]["__sha1__"])
    return {
        "train_x_val": len(tr_sha & va_sha),
        "train_x_test": len(tr_sha & te_sha),
        "val_x_test": len(va_sha & te_sha),
    }


def _describe_split_large(df: pd.DataFrame) -> dict:
    kp_vals = df[KEYPOINT_COLS_LARGE].to_numpy(dtype=float)
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_nan_vals": int(np.isnan(kp_vals).sum()),
        "frac_nan_vals": float(np.isnan(kp_vals).mean()) if kp_vals.size else 0.0,
        "n_unique_sha1": int(df["__sha1__"].nunique()),
    }


def _main_build_large_dedup():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    t0 = datetime.now()
    print(f"[start ] {t0}")

    print("[read  ] data_large splits:")
    splits_raw = _load_large_splits()
    for k, df in splits_raw.items():
        print(f"         {k}: {len(df)} rows  |  {df.shape[1]} cols  |  {SRC[k]}")

    # Pre-dedup cross-split overlap (for the report)
    for k, df in splits_raw.items():
        df["__sha1__"] = sha1_image_column(df["Image"])
    pre_overlap = {
        "train_x_val": len(
            set(splits_raw["train"]["__sha1__"]) & set(splits_raw["val"]["__sha1__"])
        ),
        "train_x_test": len(
            set(splits_raw["train"]["__sha1__"]) & set(splits_raw["test"]["__sha1__"])
        ),
        "val_x_test": len(
            set(splits_raw["val"]["__sha1__"]) & set(splits_raw["test"]["__sha1__"])
        ),
    }
    print(f"[pre   ] cross-split SHA1 overlap (pair counts): {pre_overlap}")

    # Apply alpha-rule
    splits_dedup, drop_counts = dedup_alpha(splits_raw)
    print(f"[dedup ] dropped from train: {drop_counts['train_dropped']}")
    print(f"[dedup ] dropped from val  : {drop_counts['val_dropped']}")
    print(
        f"[dedup ] dropped from test : {drop_counts['test_dropped']} (test protected)"
    )

    # Invariant check
    post_overlap = verify_no_residual_overlap(splits_dedup)
    print(f"[verify] post-dedup cross-split overlap: {post_overlap}")
    assert all(v == 0 for v in post_overlap.values()), (
        f"Residual overlap after dedup: {post_overlap}"
    )

    # Per-split stats
    stats = {k: _describe_split_large(df) for k, df in splits_dedup.items()}
    for k, s in stats.items():
        print(f"[stat  ] {k}: {s}")

    # Write outputs (drop internal __sha1__)
    for k, df in splits_dedup.items():
        out_df = df.drop(columns=["__sha1__"])
        out_path = OUT_TEMPLATE.format(split=k)
        out_df.to_csv(out_path, index=False)
        print(f"[write ] {out_path}  ({len(out_df)} rows, {out_df.shape[1]} cols)")

    # Report
    report = {
        "created_at": datetime.now().isoformat(),
        "src": SRC,
        "out_dir": CLEAN_DIR,
        "keypoint_cols": KEYPOINT_COLS_LARGE,
        "pre_dedup_overlap": pre_overlap,
        "drop_counts": drop_counts,
        "post_dedup_overlap": post_overlap,
        "per_split": stats,
        "alpha_rule": "protect test > val > train",
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ok    ] report -> {REPORT_PATH}")

    t1 = datetime.now()
    print(f"[done  ] elapsed: {t1 - t0}")
