"""Analysis / augmentation for the facial keypoint study."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from .. import augmentation as aug
from ..analysis.style import apply_style
from ..augmentation import (
    STRATEGY_FNS,
    build_hflip_index_map,
    e1_erase,
    g1_hflip,
    g2_rotate,
    g3_translate,
    g4_scale,
    p1_brightness,
    p2_contrast,
    p3_gauss_noise,
    p4_gauss_blur,
)
from ..cohorts import COHORT_G1_P, COHORT_G2_THETA, COHORT_RETRY, compute_profile_flags
from ..paths import ANALYSIS_DATA_DIR as DATA_ROOT
from ..paths import FIGURE_DIR as FIG_DIR
from ..schemas import SMALL_COLS


def _load_sample(ds_csv, idx):
    df = pd.read_csv(ds_csv)
    row = df.iloc[idx]
    img_str = row["Image"]
    img = np.array(img_str.split(), dtype=np.float32).reshape(96, 96) / 255.0
    kp_cols = [c for c in SMALL_COLS if c in df.columns]
    kp_full = np.zeros(30, dtype=np.float32)
    for i, c in enumerate(SMALL_COLS):
        if c in df.columns and not pd.isna(row[c]):
            kp_full[i] = row[c]
        else:
            kp_full[i] = np.nan
    return img, kp_full


def _find_sample(ds_csv, require_all=True):
    """Find a sample in ds_csv with no NaN in any keypoint (useful for viz)."""
    df = pd.read_csv(ds_csv)
    for idx in range(len(df)):
        row = df.iloc[idx]
        vals = [row[c] for c in SMALL_COLS if c in df.columns]
        if not require_all or not any(pd.isna(v) for v in vals):
            return idx
    return 0


def draw_face(
    ax,
    img,
    kp,
    title=None,
    dotsize=14,
    dotcolor="#E69F00",
    edgecolor="#111",
    show_pairs=False,
    kp_alpha=0.95,
    title_color="#222",
):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 95.5)
    ax.set_ylim(95.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    if kp is not None:
        for i in range(0, len(kp), 2):
            x, y = kp[i], kp[i + 1]
            if np.isnan(x) or np.isnan(y):
                continue
            ax.scatter(
                [x],
                [y],
                s=dotsize,
                color=dotcolor,
                edgecolor=edgecolor,
                linewidth=0.5,
                alpha=kp_alpha,
                zorder=5,
            )
    if title:
        pass


def fig_aug_atoms():
    rng = np.random.default_rng(7)
    # use an easy frontal sample with all 15 keypoints present
    small_train = os.path.join(DATA_ROOT, "data_full_clean_dedup_train.csv")
    idx = _find_sample(small_train, require_all=True)
    img, kp = _load_sample(small_train, idx)

    swap_pairs, self_mirror = build_hflip_index_map(SMALL_COLS)

    # one row per atom: original left, augmented right
    atoms = [
        (
            "G1  hflip  +  L↔R swap",
            lambda im, k: g1_hflip(im, k, swap_pairs, self_mirror),
            "#0072B2",
        ),
        (
            "G2  rotate  θ = +10°",
            lambda im, k: g2_rotate(im, k, 10.0) or (im, k),
            "#0072B2",
        ),
        (
            "G3  translate  (tx, ty) = (+6, −4)",
            lambda im, k: g3_translate(im, k, 6.0, -4.0) or (im, k),
            "#0072B2",
        ),
        (
            "G4  scale  s = 1.08",
            lambda im, k: g4_scale(im, k, 1.08) or (im, k),
            "#0072B2",
        ),
        (
            "P1  brightness  δ = +0.12",
            lambda im, k: (p1_brightness(im, 0.12), k),
            "#D55E00",
        ),
        ("P2  contrast  α = 1.15", lambda im, k: (p2_contrast(im, 1.15), k), "#D55E00"),
        (
            "P3  Gaussian noise  σ = 0.02",
            lambda im, k: (p3_gauss_noise(im, 0.02), k),
            "#D55E00",
        ),
        (
            "P4  Gaussian blur  σ = 0.8",
            lambda im, k: (p4_gauss_blur(im, 5, 0.8), k),
            "#D55E00",
        ),
        (
            "E1  random erase  (valid)",
            lambda im, k: (e1_erase(im, k, 10, 16), k),
            "#CC79A7",
        ),
    ]

    nrows = 3
    ncols = 6  # 3 pairs per row
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10.2, 5.6), gridspec_kw={"wspace": 0.12, "hspace": 0.35}
    )

    for k, (name, fn, color) in enumerate(atoms):
        r = k // 3
        cc = (k % 3) * 2
        ax_o, ax_a = axes[r, cc], axes[r, cc + 1]
        # ensure random draws are reproducible per atom
        np.random.seed(17 + k)
        try:
            out = fn(img.copy(), kp.copy())
            img_a, kp_a = out
        except Exception:
            img_a, kp_a = img, kp
        draw_face(ax_o, img, kp, title="original", title_color="#555")
        draw_face(ax_a, img_a, kp_a, title=name, title_color=color)

    # tidy unused cells if any
    for k in range(len(atoms) * 2, nrows * ncols):
        r = k // ncols
        cc = k % ncols
        axes[r, cc].set_visible(False)

    out = os.path.join(FIG_DIR, "F12_aug_atoms.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F12 -> {out}")


def fig_hflip_swap():
    small_train = os.path.join(DATA_ROOT, "data_full_clean_dedup_train.csv")
    idx = _find_sample(small_train, require_all=True)
    img, kp = _load_sample(small_train, idx)
    swap_pairs, self_mirror = build_hflip_index_map(SMALL_COLS)
    img2, kp2 = g1_hflip(img, kp, swap_pairs, self_mirror)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 4.0),
        gridspec_kw={"width_ratios": [1, 1, 1.25], "wspace": 0.15},
    )

    # Colour keypoints by side (left=blue, right=orange, centre=grey)
    def _kp_colors():
        # Build (name_without_xy, idx_x, idx_y, side) tuples
        colours = np.full(len(kp), "#888", dtype=object)
        for (ixL, iyL), (ixR, iyR) in swap_pairs:
            colours[ixL] = "#0072B2"
            colours[iyL] = "#0072B2"
            colours[ixR] = "#D55E00"
            colours[iyR] = "#D55E00"
        return colours

    kpc = _kp_colors()

    for ax_i, (im, kk, title) in enumerate(
        [
            (img, kp, "original"),
            (img2, kp2, "after  hflip  (pixels mirrored only)"),
        ]
    ):
        ax = axes[ax_i]
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        for i in range(0, len(kk), 2):
            ax.scatter(
                [kk[i]],
                [kk[i + 1]],
                s=30,
                color=kpc[i],
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # The correct post-hflip: keypoint L/R labels must also be swapped
    ax3 = axes[2]
    ax3.imshow(img2, cmap="gray", vmin=0, vmax=1)
    # For the corrected panel, colour reflects the TRUE side after swap
    # (after our swap step, kp2[ixL] now holds what WAS right-side; colour should be blue since it is now the 'left_*' slot)
    for (ixL, iyL), (ixR, iyR) in swap_pairs:
        # left slot (blue), right slot (orange)
        ax3.scatter(
            [kp2[ixL]],
            [kp2[iyL]],
            s=34,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        ax3.scatter(
            [kp2[ixR]],
            [kp2[iyR]],
            s=34,
            color="#D55E00",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        # arrow from left->right pair showing swap
        arrow = FancyArrowPatch(
            (kp2[ixL], kp2[iyL]),
            (kp2[ixR], kp2[iyR]),
            arrowstyle="<|-|>",
            mutation_scale=7,
            linewidth=0.7,
            color="#555",
            alpha=0.55,
        )
        ax3.add_patch(arrow)
    for i in self_mirror:
        ax3.scatter(
            [kp2[i]],
            [kp2[i + 1]],
            s=30,
            color="#888",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
    ax3.set_xticks([])
    ax3.set_yticks([])
    for s in ax3.spines.values():
        s.set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#0072B2", label="left-side keypoint"),
        mpatches.Patch(color="#D55E00", label="right-side keypoint"),
        mpatches.Patch(color="#888", label="self-mirror (nose, lip centres)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8.5,
    )

    out = os.path.join(FIG_DIR, "F13_hflip_swap.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F13 -> {out}")


def fig_cohort_pipeline():
    """Find three exemplar samples (frontal, asymmetric, hard-profile-ish)
    using inter-eye-dx as a proxy, then show them side by side with their
    cohort-conditional augmentation parameters."""
    # data_large is the richest labelled source for cohort exemplars:
    # 23 hard-profile and 499 asymmetric samples under EDA thresholds
    # (data_full collapses to ~0 hard because its profile cutoff is tighter).
    csv = os.path.join(DATA_ROOT, "data_large_clean_dedup_train.csv")
    df = pd.read_csv(csv)
    has_eyes = ~df["left_eye_center_x"].isna() & ~df["right_eye_center_x"].isna()
    df = df[has_eyes].reset_index(drop=True)

    # Use EDA-calibrated flags (same logic as training pipeline).
    is_asym, is_hard = compute_profile_flags(df, "data_large")

    # Auxiliary signals for picking the *most typical* exemplar per cohort.
    dx = (df["left_eye_center_x"] - df["right_eye_center_x"]).abs()
    mid_eye_x = (df["left_eye_center_x"] + df["right_eye_center_x"]) / 2.0
    nose_x = df["nose_tip_x"] if "nose_tip_x" in df.columns else mid_eye_x
    asym_px = (mid_eye_x - nose_x).abs()

    # Frontal: neither flag, wide eyes, small asymmetry.
    frontal_mask = (~is_asym) & (~is_hard) & asym_px.notna()
    frontal_score = dx - 3.0 * asym_px
    idx_frontal = frontal_score.where(frontal_mask).idxmax()

    # Asymmetric: flagged asym but NOT hard; max face_asymmetry inside that set.
    asym_only_mask = is_asym & (~is_hard) & asym_px.notna()
    idx_asym = asym_px.where(asym_only_mask).idxmax()

    # Hard profile: flagged hard; most extreme asymmetry, smallest eye-dx.
    hard_mask = is_hard & asym_px.notna()
    hard_score = asym_px - 0.5 * dx
    idx_hard = hard_score.where(hard_mask).idxmax()

    samples = [
        (
            "Frontal",
            idx_frontal,
            dict(
                g1_p=COHORT_G1_P["frontal"],
                theta=COHORT_G2_THETA["frontal"],
                retry=COHORT_RETRY["frontal"],
            ),
            "#009E73",
        ),
        (
            "Asymmetric",
            idx_asym,
            dict(
                g1_p=COHORT_G1_P["asym"],
                theta=COHORT_G2_THETA["asym"],
                retry=COHORT_RETRY["asym"],
            ),
            "#E69F00",
        ),
        (
            "Hard profile",
            idx_hard,
            dict(
                g1_p=COHORT_G1_P["hard"],
                theta=COHORT_G2_THETA["hard"],
                retry=COHORT_RETRY["hard"],
            ),
            "#D55E00",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), gridspec_kw={"wspace": 0.18})

    for k, (name, ridx, params, color) in enumerate(samples):
        row = df.iloc[ridx]
        img_str = row["Image"]
        img = np.array(img_str.split(), dtype=np.float32).reshape(96, 96) / 255.0
        kp = np.zeros(30, dtype=np.float32)
        for i, c in enumerate(SMALL_COLS):
            if c in df.columns and not pd.isna(row[c]):
                kp[i] = row[c]
            else:
                kp[i] = np.nan
        ax_img = axes[k]
        draw_face(ax_img, img, kp, dotsize=22)
        ax_img.set_title(name, fontsize=14, fontweight="bold", color=color, pad=6)

    out = os.path.join(FIG_DIR, "F14_cohort_pipeline.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F14 -> {out}")


def fig_strategy_grid():
    csv = os.path.join(DATA_ROOT, "data_full_clean_dedup_train.csv")
    idx = _find_sample(csv, require_all=True)
    img, kp = _load_sample(csv, idx)
    swap_pairs, self_mirror = build_hflip_index_map(SMALL_COLS)

    strategies = ["S1", "S2", "S3", "S4", "S5"]
    strat_desc = {
        "S1": "photometric only  (P1+P2+P3)",
        "S2": "geometric only  (G1+G2+G3)",
        "S3": "S1 + S2",
        "S4": "S3 + G4 (scale) + P4 (blur)",
        "S5": "S3 + E1 (random erase)",
    }

    ncols = 5  # 1 original + 4 augmented samples
    fig, axes = plt.subplots(
        len(strategies),
        ncols,
        figsize=(10.4, 10.0),
        gridspec_kw={"wspace": 0.08, "hspace": 0.3},
    )

    for r, s in enumerate(strategies):
        # Col 0: original with strategy label
        axes[r, 0].axis("off")
        axes[r, 0].text(
            0.0,
            0.62,
            s,
            transform=axes[r, 0].transAxes,
            fontsize=16,
            fontweight="bold",
            color="#222",
        )
        axes[r, 0].text(
            0.0,
            0.35,
            strat_desc[s],
            transform=axes[r, 0].transAxes,
            fontsize=8.5,
            color="#555",
            style="italic",
            wrap=True,
        )
        compose = STRATEGY_FNS[s]
        for c in range(1, ncols):
            np.random.seed(100 * r + c)
            img_a, kp_a = compose(
                img.copy(), kp.copy(), swap_pairs, self_mirror, "frontal"
            )
            draw_face(axes[r, c], img_a, kp_a)

    # Header labels over columns 1..4
    for c in range(1, ncols):
        axes[0, c].set_title(f"random draw #{c}", fontsize=8.5, color="#555", pad=4)

    out = os.path.join(FIG_DIR, "F15_strategy_grid.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F15 -> {out}")


def fig_probability_map():
    # Two panels:
    #  (a) bar chart of per-atom unconditional probability (frontal cohort defaults)
    #  (b) grouped bars of cohort-conditional G1/G2 (p and theta)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 3.6),
        gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.28},
    )

    # panel a
    ax = axes[0]
    atoms = [
        ("G1\nhflip", aug.P_G1, "#0072B2", "geometric"),
        ("G2\nrotate", aug.P_G2, "#0072B2", "geometric"),
        ("G3\ntranslate", aug.P_G3, "#0072B2", "geometric"),
        ("G4\nscale", aug.P_G4, "#009E73", "geo+"),
        ("P1\nbrightness", aug.P_P1, "#D55E00", "photometric"),
        ("P2\ncontrast", aug.P_P2, "#D55E00", "photometric"),
        ("P3\nnoise", aug.P_P3, "#D55E00", "photometric"),
        ("P4\nblur", aug.P_P4, "#E69F00", "photo+"),
        ("E1\nerase", aug.P_E1, "#CC79A7", "erasing"),
    ]
    xs = np.arange(len(atoms))
    bars = ax.bar(
        xs,
        [a[1] for a in atoms],
        width=0.62,
        color=[a[2] for a in atoms],
        edgecolor="white",
        linewidth=0.6,
    )
    for b, (lab, p, col, _) in zip(bars, atoms):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.015,
            f"{p:.2f}",
            ha="center",
            fontsize=8,
            color="#222",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([a[0] for a in atoms], fontsize=8)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Application probability per sample")
    # side-band legend
    from matplotlib.lines import Line2D

    groups = [
        ("geometric", "#0072B2"),
        ("geo  (scale)", "#009E73"),
        ("photometric", "#D55E00"),
        ("photo (blur)", "#E69F00"),
        ("erasing", "#CC79A7"),
    ]
    handles = [Line2D([0], [0], color=c, linewidth=5) for _, c in groups]
    ax.legend(
        handles, [g for g, _ in groups], loc="upper right", fontsize=7.5, frameon=False
    )

    # panel b: cohort-dependent G1/G2
    ax = axes[1]
    cohorts = ["frontal", "asym", "hard"]
    cx = np.arange(len(cohorts))
    w = 0.36
    g1_vals = [COHORT_G1_P[c] for c in cohorts]
    theta_vals = [COHORT_G2_THETA[c] for c in cohorts]

    b1 = ax.bar(
        cx - w / 2,
        g1_vals,
        w,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.6,
        label=r"$p_{G1}$ (hflip)",
    )
    # normalise theta to [0,1] for co-plotting, with a secondary axis
    ax2 = ax.twinx()
    b2 = ax2.bar(
        cx + w / 2,
        theta_vals,
        w,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.6,
        label=r"$\theta_{G2}^{\max}$  (deg)",
    )
    ax.set_xticks(cx)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel(r"$p_{G1}$  (hflip probability)", color="#0072B2")
    ax2.set_ylabel(r"$\theta_{G2}^{\max}$  (degrees)", color="#D55E00")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax.set_ylim(0, 0.65)
    ax2.set_ylim(0, 14)
    for b, v in zip(b1, g1_vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.015,
            f"{v:.2f}",
            ha="center",
            fontsize=8,
            color="#0072B2",
        )
    for b, v in zip(b2, theta_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 0.3,
            f"{v:.0f}°",
            ha="center",
            fontsize=8,
            color="#D55E00",
        )

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    out = os.path.join(FIG_DIR, "F16_probability_map.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F16 -> {out}")


def _main_build_aug_figures():
    apply_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_aug_atoms()
    fig_hflip_swap()
    fig_cohort_pipeline()
    fig_strategy_grid()
    fig_probability_map()
    print(f"\nAug figures -> {FIG_DIR}")
