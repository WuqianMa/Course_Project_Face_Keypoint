"""Data prep / full for the facial keypoint study."""

import hashlib
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from ..cohorts import get_profile_config
from ..paths import CLEAN_DIR as OUT_DIR
from ..paths import RAW_CSV

SEED = 42

KEYPOINT_COLS = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "left_eye_inner_corner_x",
    "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",
    "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",
    "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",
    "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",
    "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",
    "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",
    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",
    "right_eyebrow_outer_end_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y",
    "mouth_center_top_lip_x",
    "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
]

assert len(KEYPOINT_COLS) == 30


def add_mask_columns(df, kp_cols):
    """Append mask_<col> = (1.0 if coord present, 0.0 if NaN) for each kp col.
    Keeps the original NaN values in kp_cols intact (no imputation).
    """
    for c in kp_cols:
        df[f"mask_{c}"] = df[c].notna().astype(np.float32)
    return df


def sha1_image_column(series):
    """SHA1 fingerprint of the raw pixel string (96*96 = 9216 ints, space-separated)."""
    return series.map(lambda s: hashlib.sha1(s.encode()).hexdigest())


def split_612(df_in, seed):
    """Exact replication of baseline_model_training.ipynb `split_612`."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df_in))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    tr = idx[:n_train]
    va = idx[n_train : n_train + n_val]
    te = idx[n_train + n_val :]
    return {
        "train": df_in.iloc[tr].reset_index(drop=True),
        "val": df_in.iloc[va].reset_index(drop=True),
        "test": df_in.iloc[te].reset_index(drop=True),
    }


def compute_profile_flags_full(df_split, cutoff_px):
    """Mirror `compute_profile_flags(df, 'data_full')` from 01_augmentation.ipynb.

    Rule: only classify samples where both left/right eye_center labels are
    present (mask bits = 1). Unlabeled samples -> is_asym=False, is_hard=False.
    """
    n = len(df_split)
    has_both_eyes = (df_split["mask_left_eye_center_x"].values > 0.5) & (
        df_split["mask_right_eye_center_x"].values > 0.5
    )
    # Missing kp values are NaN here; np.abs(NaN - NaN) = NaN
    inter_eye_dx = np.abs(
        df_split["left_eye_center_x"].values - df_split["right_eye_center_x"].values
    )
    # Guard: inter_eye_dx may contain NaN for rows where eyes are unlabeled
    # but has_both_eyes masks them out anyway. Force NaN -> False comparison.
    with np.errstate(invalid="ignore"):
        is_asym = has_both_eyes & (inter_eye_dx < cutoff_px)
    is_hard = np.zeros(n, dtype=bool)
    return is_asym.astype(bool), is_hard.astype(bool)


def _describe_split_full(df, name):
    """Compact per-split diagnostics for the rebuild report."""
    kp_vals = df[KEYPOINT_COLS].to_numpy(dtype=float)
    n_vals = kp_vals.size
    return {
        "n_rows": int(len(df)),
        "n_nan_vals": int(np.isnan(kp_vals).sum()),
        "frac_nan_vals": float(np.isnan(kp_vals).mean()),
        "n_zero_vals": int((kp_vals == 0.0).sum()),
        "frac_zero_vals": float((kp_vals == 0.0).mean()),
        "n_is_asym": int(df["is_asym"].sum()),
        "frac_is_asym": float(df["is_asym"].mean()),
        "n_is_hard": int(df["is_hard"].sum()),
        "frac_is_hard": float(df["is_hard"].mean()),
        "n_unique_sha1": int(df["__sha1__"].nunique()),
    }


def dedup_across_splits(splits):
    """Option alpha: protect test > val > train.
    - Drop train rows whose SHA1 appears in val or test.
    - Drop val rows whose SHA1 appears in test.
    - Test is untouched.
    Returns dict with the cleaned splits and per-step drop counts.
    """
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

    return {
        "train": tr_clean,
        "val": va_clean,
        "test": splits["test"],
    }, {"train_dropped": tr_drop, "val_dropped": va_drop, "test_dropped": 0}


def _main_rebuild_splits():
    os.makedirs(OUT_DIR, exist_ok=True)
    PROFILE_CUTOFF_PX = float(get_profile_config()["data_small"]["profile_cutoff_px"])
    t0 = datetime.now()
    print(f"[start] {t0}")
    print(f"[read ] {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    n_all = len(df)
    n_nan = int(df[KEYPOINT_COLS].isna().sum().sum())
    n_zero = int((df[KEYPOINT_COLS].to_numpy(dtype=float) == 0.0).sum())
    print(
        f"       {n_all} rows | NaN coord vals: {n_nan} ({n_nan / (n_all * 30):.4f}) "
        f"| exact-0 coord vals: {n_zero}"
    )
    assert "Image" in df.columns, "Raw training.csv missing Image column"

    # 1) keep NaN, add mask columns
    df = add_mask_columns(df, KEYPOINT_COLS)
    # 2) pixel fingerprint
    df["__sha1__"] = sha1_image_column(df["Image"])
    n_unique_all = int(df["__sha1__"].nunique())
    print(
        f"[hash ] unique images (by SHA1): {n_unique_all} / {n_all} "
        f"({n_all - n_unique_all} within-raw duplicate rows)"
    )

    # 3) random 60/20/20 split, SEED=42 (reproduce legacy identity)
    splits_v1 = split_612(df, seed=SEED)
    for k, v in splits_v1.items():
        print(f"       non-dedup {k}: {len(v)} rows")

    # 4) cohort flags on non-dedup splits
    for k, part in splits_v1.items():
        is_asym, is_hard = compute_profile_flags_full(part, PROFILE_CUTOFF_PX)
        part["is_asym"] = is_asym
        part["is_hard"] = is_hard

    # 5) dedup variant (option alpha: test > val > train)
    splits_v2, drop_counts = dedup_across_splits(splits_v1)
    print(
        f"[dedup] dropped from train: {drop_counts['train_dropped']} rows "
        f"(duplicates of val or test)"
    )
    print(
        f"[dedup] dropped from val:   {drop_counts['val_dropped']} rows "
        f"(duplicates of test)"
    )

    for k in ("train", "val", "test"):
        print(f"       dedup {k}: {len(splits_v2[k])} rows")

    # 6) write outputs - drop the internal __sha1__ column before CSV
    report = {
        "source": RAW_CSV,
        "seed": SEED,
        "profile_cutoff_px": PROFILE_CUTOFF_PX,
        "n_raw_rows": n_all,
        "n_unique_sha1_in_raw": n_unique_all,
        "nan_frac_in_raw": n_nan / (n_all * 30),
        "exact_zero_count_in_raw": n_zero,
        "variants": {},
    }

    for variant, splits in (("nondedup", splits_v1), ("dedup", splits_v2)):
        variant_stats = {}
        for k, part in splits.items():
            variant_stats[k] = _describe_split_full(part, k)
            # Drop helper col before writing
            out_df = part.drop(columns=["__sha1__"])
            if variant == "nondedup":
                out_path = os.path.join(OUT_DIR, f"data_full_clean_{k}.csv")
            else:
                out_path = os.path.join(OUT_DIR, f"data_full_clean_dedup_{k}.csv")
            out_df.to_csv(out_path, index=False)
            print(
                f"[write] {out_path}  ({len(out_df)} rows, {len(out_df.columns)} cols)"
            )
        report["variants"][variant] = variant_stats

    report["dedup_drop_counts"] = drop_counts

    # 7) save report
    report_path = os.path.join(OUT_DIR, "rebuild_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ok  ] report -> {report_path}")

    t1 = datetime.now()
    print(f"[done ] elapsed: {t1 - t0}")
