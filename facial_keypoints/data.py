"""Data for the facial keypoint study."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentation import build_hflip_index_map
from .cohorts import _classify_cohort, compute_profile_flags
from .paths import CLEAN_DIR as _CLEAN_DIR
from .paths import DATA_DIR as _DATA_DIR
from .schemas import LARGE_COLS, SMALL_COLS

DATASET_REGISTRY = {
    "small_raw": {
        "dir": _DATA_DIR,
        "prefix": "data_small",
        "ds_name": "data_small",
        "has_mask": False,
    },
    "large_raw": {
        "dir": _DATA_DIR,
        "prefix": "data_large",
        "ds_name": "data_large",
        "has_mask": False,
    },
    "large_dedup": {
        "dir": _CLEAN_DIR,
        "prefix": "data_large_clean_dedup",
        "ds_name": "data_large",
        "has_mask": False,
    },
    "full_raw_clean": {
        "dir": _CLEAN_DIR,
        "prefix": "data_full_clean",
        "ds_name": "data_full",
        "has_mask": True,
    },
    "full_dedup": {
        "dir": _CLEAN_DIR,
        "prefix": "data_full_clean_dedup",
        "ds_name": "data_full",
        "has_mask": True,
    },
}

TARGET_COLS_SMALL = list(SMALL_COLS)

assert len(TARGET_COLS_SMALL) == 30

TARGET_COLS_LARGE = list(LARGE_COLS)

assert len(TARGET_COLS_LARGE) == 8

TARGET_COLS_FULL = TARGET_COLS_SMALL


def default_target_cols(ds_name):
    if ds_name == "data_small":
        return TARGET_COLS_SMALL
    if ds_name == "data_large":
        return TARGET_COLS_LARGE
    if ds_name == "data_full":
        return TARGET_COLS_FULL
    raise ValueError(f"unknown ds_name: {ds_name}")


def parse_image_column(series):
    arr = np.empty((len(series), 96, 96), dtype=np.float32)
    for i, s in enumerate(series.values):
        arr[i] = np.asarray(s.split(), dtype=np.float32).reshape(96, 96)
    arr /= 255.0
    return arr


def build_tensors(df_split, target_cols, ds_name=None):
    X = parse_image_column(df_split["Image"])
    y = df_split[target_cols].to_numpy(dtype=np.float32)
    y_norm = (y - 48.0) / 48.0
    X_t = torch.from_numpy(X).unsqueeze(1)
    y_t = torch.from_numpy(y_norm)
    if ds_name == "data_full":
        mask_cols = [f"mask_{c}" for c in target_cols]
        m = df_split[mask_cols].to_numpy(dtype=np.float32)
        m_t = torch.from_numpy(m)
        # Masked-loss invariant: NaN must be sanitized before reaching loss.
        y_t = torch.nan_to_num(y_t, nan=0.0)
        is_asym, is_hard = compute_profile_flags(df_split, ds_name)
        return X_t, y_t, is_asym, is_hard, m_t
    if ds_name is not None:
        is_asym, is_hard = compute_profile_flags(df_split, ds_name)
        return X_t, y_t, is_asym, is_hard
    return X_t, y_t


class KeypointAugDataset(Dataset):
    """Per-sample augmented keypoint dataset with HFlip mask-sync."""

    def __init__(
        self,
        X_tensor,
        y_tensor,
        transform_fn=None,
        target_cols=None,
        is_asym=None,
        is_hard=None,
        m_tensor=None,
        hflip_mask_sync=True,
    ):
        self.X = X_tensor
        self.y = y_tensor
        self.m = m_tensor
        self.transform_fn = transform_fn
        self.hflip_mask_sync = bool(hflip_mask_sync)
        self.is_asym = (
            is_asym if is_asym is not None else np.zeros(len(X_tensor), dtype=bool)
        )
        self.is_hard = (
            is_hard if is_hard is not None else np.zeros(len(X_tensor), dtype=bool)
        )
        if transform_fn is not None:
            assert target_cols is not None, (
                "target_cols required when transform_fn is given"
            )
            self.swap_pairs, self.self_mirror_x = build_hflip_index_map(target_cols)
        else:
            self.swap_pairs, self.self_mirror_x = [], []

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.transform_fn is None:
            if self.m is not None:
                return self.X[i], self.y[i], self.m[i]
            return self.X[i], self.y[i]

        img = self.X[i, 0].numpy()
        kp_norm = self.y[i].numpy()
        kp_px = kp_norm * 48.0 + 48.0
        cohort = _classify_cohort(bool(self.is_asym[i]), bool(self.is_hard[i]))
        kp_pre = kp_px.copy()
        img_aug, kp_aug = self.transform_fn(
            img,
            kp_px,
            self.swap_pairs,
            self.self_mirror_x,
            cohort=cohort,
        )
        kp_back = ((kp_aug - 48.0) / 48.0).astype(np.float32)
        img_t = torch.from_numpy(np.ascontiguousarray(img_aug)).unsqueeze(0).float()

        if self.m is None:
            return img_t, torch.from_numpy(kp_back)

        m_i = self.m[i].clone()
        if self.hflip_mask_sync and len(self.swap_pairs) > 0:
            # Detect HFlip via slot-swap voting across LR pairs (robust to
            # rotation/translation/scale noise from sibling atoms).
            votes_swap = 0
            votes_noswap = 0
            for (ix_L, iy_L), (ix_R, iy_R) in self.swap_pairs:
                x_post = kp_aug[ix_L]
                d_swap = abs(x_post - (96.0 - kp_pre[ix_R]))
                d_noswap = abs(x_post - kp_pre[ix_L])
                if d_swap < d_noswap:
                    votes_swap += 1
                else:
                    votes_noswap += 1
            flipped = votes_swap > votes_noswap
            if flipped:
                for (ix_L, iy_L), (ix_R, iy_R) in self.swap_pairs:
                    (m_i[ix_L], m_i[iy_L], m_i[ix_R], m_i[iy_R]) = (
                        m_i[ix_R].clone(),
                        m_i[iy_R].clone(),
                        m_i[ix_L].clone(),
                        m_i[iy_L].clone(),
                    )

        return img_t, torch.from_numpy(kp_back), m_i


def make_loaders(
    dataset_name,
    target_cols=None,
    transform_fn=None,
    batch_size=128,
    num_workers=4,
    hflip_mask_sync=True,
):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"unknown dataset_name: {dataset_name!r}. valid: {list(DATASET_REGISTRY)}"
        )
    reg = DATASET_REGISTRY[dataset_name]
    ds_name = reg["ds_name"]
    data_dir = reg["dir"]
    prefix = reg["prefix"]

    if target_cols is None:
        target_cols = default_target_cols(ds_name)

    path_tr = os.path.join(data_dir, f"{prefix}_train.csv")
    path_va = os.path.join(data_dir, f"{prefix}_val.csv")
    tr = pd.read_csv(path_tr)
    va = pd.read_csv(path_va)

    if ds_name == "data_full":
        Xtr, ytr, tr_asym, tr_hard, mtr = build_tensors(
            tr, target_cols, ds_name=ds_name
        )
        Xva, yva, va_asym, va_hard, mva = build_tensors(
            va, target_cols, ds_name=ds_name
        )
        ds_tr = KeypointAugDataset(
            Xtr,
            ytr,
            transform_fn=transform_fn,
            target_cols=target_cols,
            is_asym=tr_asym,
            is_hard=tr_hard,
            m_tensor=mtr,
            hflip_mask_sync=hflip_mask_sync,
        )
        ds_va = KeypointAugDataset(
            Xva,
            yva,
            transform_fn=None,
            m_tensor=mva,
            hflip_mask_sync=hflip_mask_sync,
        )
    else:
        Xtr, ytr, tr_asym, tr_hard = build_tensors(tr, target_cols, ds_name=ds_name)
        Xva, yva, va_asym, va_hard = build_tensors(va, target_cols, ds_name=ds_name)
        ds_tr = KeypointAugDataset(
            Xtr,
            ytr,
            transform_fn=transform_fn,
            target_cols=target_cols,
            is_asym=tr_asym,
            is_hard=tr_hard,
            hflip_mask_sync=hflip_mask_sync,
        )
        ds_va = KeypointAugDataset(
            Xva,
            yva,
            transform_fn=None,
            hflip_mask_sync=hflip_mask_sync,
        )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    n_tr = len(ds_tr)
    n_va = len(ds_va)
    tr_cohort_frac = {
        "frontal": float(((~tr_asym) & (~tr_hard)).sum() / n_tr),
        "asym": float((tr_asym & (~tr_hard)).sum() / n_tr),
        "hard": float(tr_hard.sum() / n_tr),
    }
    val_flags = {"is_asym": va_asym, "is_hard": va_hard}
    return dl_tr, dl_va, n_tr, n_va, tr_cohort_frac, val_flags
