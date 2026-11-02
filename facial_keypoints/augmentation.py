"""Augmentation for the facial keypoint study."""

import math

import numpy as np
import torch

from .cohorts import COHORT_G1_P, COHORT_G2_THETA, COHORT_RETRY
from .schemas import SMALL_COLS

LR_BASE_PAIRS = [
    ("left_eye_center", "right_eye_center"),
    ("left_eye_inner_corner", "right_eye_inner_corner"),
    ("left_eye_outer_corner", "right_eye_outer_corner"),
    ("left_eyebrow_inner_end", "right_eyebrow_inner_end"),
    ("left_eyebrow_outer_end", "right_eyebrow_outer_end"),
    ("mouth_left_corner", "mouth_right_corner"),
]

SELF_MIRROR_BASE = ["nose_tip", "mouth_center_top_lip", "mouth_center_bottom_lip"]

FULL_COLS = list(SMALL_COLS)


def build_hflip_index_map(target_cols):
    col_to_idx = {c: i for i, c in enumerate(target_cols)}
    swap_pairs = []
    for L, R in LR_BASE_PAIRS:
        lx, ly, rx, ry = f"{L}_x", f"{L}_y", f"{R}_x", f"{R}_y"
        if lx in col_to_idx and rx in col_to_idx:
            swap_pairs.append(
                (
                    (col_to_idx[lx], col_to_idx[ly]),
                    (col_to_idx[rx], col_to_idx[ry]),
                )
            )
    self_mirror = []
    for name in SELF_MIRROR_BASE:
        nx = f"{name}_x"
        if nx in col_to_idx:
            self_mirror.append(col_to_idx[nx])
    return swap_pairs, self_mirror


def _within_bounds(kp, lo=0.0, hi=95.0):
    return np.all((kp >= lo) & (kp <= hi))


def g1_hflip(img, kp, swap_pairs, self_mirror_x):
    img_out = img[:, ::-1].copy()
    kp_out = kp.copy()
    kp_out[0::2] = 95.0 - kp_out[0::2]
    for (ixL, iyL), (ixR, iyR) in swap_pairs:
        kp_out[ixL], kp_out[ixR] = kp_out[ixR], kp_out[ixL]
        kp_out[iyL], kp_out[iyR] = kp_out[iyR], kp_out[iyL]
    return img_out, kp_out


def _rotate_image(img, theta_deg):
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    theta = math.radians(theta_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    mat = torch.tensor(
        [[cos, sin, 0.0], [-sin, cos, 0.0]], dtype=torch.float32
    ).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(mat, t.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(
        t, grid, align_corners=False, padding_mode="border"
    )
    return out.squeeze().numpy()


def g2_rotate(img, kp, theta_deg):
    cx = cy = 47.5
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    kp_out = kp.copy()
    xs = kp_out[0::2] - cx
    ys = kp_out[1::2] - cy
    kp_out[0::2] = xs * c - ys * s + cx
    kp_out[1::2] = xs * s + ys * c + cy
    if not _within_bounds(kp_out):
        return None
    img_out = _rotate_image(img, theta_deg)
    return img_out, kp_out


def _translate_image(img, tx, ty):
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    dx = -2.0 * tx / 96.0
    dy = -2.0 * ty / 96.0
    mat = torch.tensor([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=torch.float32).unsqueeze(
        0
    )
    grid = torch.nn.functional.affine_grid(mat, t.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(
        t, grid, align_corners=False, padding_mode="border"
    )
    return out.squeeze().numpy()


def g3_translate(img, kp, tx, ty):
    kp_out = kp.copy()
    kp_out[0::2] += tx
    kp_out[1::2] += ty
    if not _within_bounds(kp_out):
        return None
    img_out = _translate_image(img, tx, ty)
    return img_out, kp_out


def _scale_image(img, s):
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    inv = 1.0 / s
    mat = torch.tensor(
        [[inv, 0.0, 0.0], [0.0, inv, 0.0]], dtype=torch.float32
    ).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(mat, t.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(
        t, grid, align_corners=False, padding_mode="border"
    )
    return out.squeeze().numpy()


def g4_scale(img, kp, s):
    cx = cy = 47.5
    kp_out = kp.copy()
    kp_out[0::2] = (kp_out[0::2] - cx) * s + cx
    kp_out[1::2] = (kp_out[1::2] - cy) * s + cy
    if not _within_bounds(kp_out):
        return None
    img_out = _scale_image(img, s)
    return img_out, kp_out


def p1_brightness(img, delta):
    return np.clip(img + delta, 0.0, 1.0)


def p2_contrast(img, alpha):
    mean = img.mean()
    return np.clip((img - mean) * alpha + mean, 0.0, 1.0)


def p3_gauss_noise(img, sigma):
    return np.clip(
        img + np.random.randn(*img.shape).astype(np.float32) * sigma, 0.0, 1.0
    )


def p4_gauss_blur(img, ksize, sigma):
    r = ksize // 2
    ax = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k /= k.sum()
    padded = np.pad(img, r, mode="edge")
    out = np.zeros_like(img)
    for i in range(ksize):
        for j in range(ksize):
            out += k[i, j] * padded[i : i + 96, j : j + 96]
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def e1_erase(img, kp, patch_min=8, patch_max=16):
    h = np.random.randint(patch_min, patch_max + 1)
    w = np.random.randint(patch_min, patch_max + 1)
    top = np.random.randint(0, 96 - h + 1)
    left = np.random.randint(0, 96 - w + 1)
    xs = kp[0::2]
    ys = kp[1::2]
    inside = (xs >= left) & (xs < left + w) & (ys >= top) & (ys < top + h)
    if inside.any():
        return img
    img_out = img.copy()
    img_out[top : top + h, left : left + w] = np.random.rand(h, w).astype(np.float32)
    return img_out


P_G1, P_G2, P_G3, P_G4 = 0.5, 0.5, 0.5, 0.3

P_P1, P_P2, P_P3, P_P4 = 0.5, 0.5, 0.3, 0.2

P_E1 = 0.25

G2_TH_MAX = 12.0

G3_T_MAX = 8.0

G4_S_LO, G4_S_HI = 0.9, 1.1

P1_DELTA = 0.15

P2_ALPHA_LO, P2_ALPHA_HI = 0.85, 1.15

P3_SIGMA_MAX = 0.02

P4_SIGMA_LO, P4_SIGMA_HI = 0.1, 1.0

AUG_TELEMETRY = {
    "frontal": 0,
    "asym": 0,
    "hard": 0,
    "g1_applied": {"frontal": 0, "asym": 0, "hard": 0},
    "g2_applied": {"frontal": 0, "asym": 0, "hard": 0},
}


def _reset_telemetry():
    AUG_TELEMETRY.clear()
    AUG_TELEMETRY.update(
        {
            "frontal": 0,
            "asym": 0,
            "hard": 0,
            "g1_applied": {"frontal": 0, "asym": 0, "hard": 0},
            "g2_applied": {"frontal": 0, "asym": 0, "hard": 0},
        }
    )


def _apply_geom_with_retry(atom_fn, img, kp, max_retry=3):
    for _ in range(max_retry):
        out = atom_fn(img, kp)
        if out is not None:
            return out
    return img, kp


def _maybe_g1(img, kp, swap_pairs, self_mirror_x, cohort="frontal"):
    p_g1 = COHORT_G1_P[cohort]
    if p_g1 > 0 and np.random.rand() < p_g1:
        AUG_TELEMETRY["g1_applied"][cohort] += 1
        return g1_hflip(img, kp, swap_pairs, self_mirror_x)
    return img, kp


def _maybe_g2(img, kp, cohort="frontal"):
    if np.random.rand() < P_G2:
        AUG_TELEMETRY["g2_applied"][cohort] += 1
        theta_max = COHORT_G2_THETA[cohort]
        retry = COHORT_RETRY[cohort]
        theta = float(np.random.uniform(-theta_max, theta_max))
        return _apply_geom_with_retry(
            lambda im, k: g2_rotate(im, k, theta),
            img,
            kp,
            max_retry=retry,
        )
    return img, kp


def _maybe_g3(img, kp):
    if np.random.rand() < P_G3:
        tx = float(np.random.uniform(-G3_T_MAX, G3_T_MAX))
        ty = float(np.random.uniform(-G3_T_MAX, G3_T_MAX))
        return _apply_geom_with_retry(
            lambda im, k: g3_translate(im, k, tx, ty), img, kp
        )
    return img, kp


def _maybe_g4(img, kp):
    if np.random.rand() < P_G4:
        s = float(np.random.uniform(G4_S_LO, G4_S_HI))
        return _apply_geom_with_retry(lambda im, k: g4_scale(im, k, s), img, kp)
    return img, kp


def _maybe_p1(img):
    if np.random.rand() < P_P1:
        d = float(np.random.uniform(-P1_DELTA, P1_DELTA))
        return p1_brightness(img, d)
    return img


def _maybe_p2(img):
    if np.random.rand() < P_P2:
        a = float(np.random.uniform(P2_ALPHA_LO, P2_ALPHA_HI))
        return p2_contrast(img, a)
    return img


def _maybe_p3(img):
    if np.random.rand() < P_P3:
        s = float(np.random.uniform(0.0, P3_SIGMA_MAX))
        return p3_gauss_noise(img, s)
    return img


def _maybe_p4(img):
    if np.random.rand() < P_P4:
        ksize = int(np.random.choice([3, 5]))
        sig = float(np.random.uniform(P4_SIGMA_LO, P4_SIGMA_HI))
        return p4_gauss_blur(img, ksize, sig)
    return img


def _maybe_e1(img, kp):
    if np.random.rand() < P_E1:
        return e1_erase(img, kp)
    return img


def compose_S1(img, kp, swap_pairs, self_mirror_x, cohort="frontal"):
    AUG_TELEMETRY[cohort] += 1
    img = _maybe_p1(img)
    img = _maybe_p2(img)
    img = _maybe_p3(img)
    return img, kp


def compose_S2(img, kp, swap_pairs, self_mirror_x, cohort="frontal"):
    AUG_TELEMETRY[cohort] += 1
    img, kp = _maybe_g1(img, kp, swap_pairs, self_mirror_x, cohort=cohort)
    img, kp = _maybe_g2(img, kp, cohort=cohort)
    img, kp = _maybe_g3(img, kp)
    return img, kp


def compose_S3(img, kp, swap_pairs, self_mirror_x, cohort="frontal"):
    AUG_TELEMETRY[cohort] += 1
    img, kp = _maybe_g1(img, kp, swap_pairs, self_mirror_x, cohort=cohort)
    img, kp = _maybe_g2(img, kp, cohort=cohort)
    img, kp = _maybe_g3(img, kp)
    img = _maybe_p1(img)
    img = _maybe_p2(img)
    img = _maybe_p3(img)
    return img, kp


def compose_S4(img, kp, swap_pairs, self_mirror_x, cohort="frontal"):
    AUG_TELEMETRY[cohort] += 1
    img, kp = _maybe_g1(img, kp, swap_pairs, self_mirror_x, cohort=cohort)
    img, kp = _maybe_g2(img, kp, cohort=cohort)
    img, kp = _maybe_g3(img, kp)
    img, kp = _maybe_g4(img, kp)
    img = _maybe_p1(img)
    img = _maybe_p2(img)
    img = _maybe_p3(img)
    img = _maybe_p4(img)
    return img, kp


def compose_S5(img, kp, swap_pairs, self_mirror_x, cohort="frontal"):
    AUG_TELEMETRY[cohort] += 1
    img, kp = _maybe_g1(img, kp, swap_pairs, self_mirror_x, cohort=cohort)
    img, kp = _maybe_g2(img, kp, cohort=cohort)
    img, kp = _maybe_g3(img, kp)
    img = _maybe_p1(img)
    img = _maybe_p2(img)
    img = _maybe_p3(img)
    img = _maybe_e1(img, kp)
    return img, kp


STRATEGY_FNS = {
    "S1": compose_S1,
    "S2": compose_S2,
    "S3": compose_S3,
    "S4": compose_S4,
    "S5": compose_S5,
}
