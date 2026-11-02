"""Analysis / detail for the facial keypoint study."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.style import (
    DATASET_LABELS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS_ORDER,
    apply_style,
    panel_label,
)
from ..paths import AGGREGATE_DIR as PAPER_DIR
from ..paths import ANALYSIS_DATA_DIR as DATA_ROOT
from ..paths import FIGURE_DIR as FIG_DIR
from ..schemas import LARGE_COLS, SMALL_COLS


def _load():
    long_df = pd.read_csv(os.path.join(PAPER_DIR, "tier1_full_long.csv"))
    epoch_df = pd.read_csv(os.path.join(PAPER_DIR, "tier1_full_epoch.csv"))
    return long_df, epoch_df


def fig_label_sparsity():
    raw_data_root = os.path.abspath(os.path.join(DATA_ROOT, "..", "Raw Data"))
    specs = [
        (
            "Small (raw)\nfavour keypoints",
            os.path.join(raw_data_root, "data_small.csv"),
            SMALL_COLS,
        ),
        (
            "Large (raw)\nfavour rows",
            os.path.join(raw_data_root, "data_large.csv"),
            LARGE_COLS,
        ),
        ("Full (raw)\nkeep both", os.path.join(raw_data_root, "data.csv"), SMALL_COLS),
    ]
    rows = []
    for name, csv, cols in specs:
        df = pd.read_csv(csv)
        pts = {}
        # collapse _x/_y pairs into keypoint name
        base_names = []
        for c in cols:
            if c.endswith("_x"):
                base_names.append(c[:-2])
        for bn in base_names:
            xc, yc = bn + "_x", bn + "_y"
            if xc in df.columns and yc in df.columns:
                present = (~df[xc].isna() & ~df[yc].isna()).mean()
                pts[bn] = present
        rows.append((name, pts, len(df)))

    # union of keypoint names, ordered by data_small schema
    all_kps = []
    for c in SMALL_COLS:
        if c.endswith("_x"):
            all_kps.append(c[:-2])

    mat = np.full((len(rows), len(all_kps)), np.nan)
    for r, (_, pts, _) in enumerate(rows):
        for ci, kp in enumerate(all_kps):
            if kp in pts:
                mat[r, ci] = pts[kp]

    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{n}\n(n={sz})" for (n, _, sz) in rows], fontsize=7.5)
    ax.set_xticks(range(len(all_kps)))
    ax.set_xticklabels(
        [k.replace("_", "\n") for k in all_kps], fontsize=6.5, rotation=0
    )
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if np.isnan(v):
                ax.text(c, r, "—", ha="center", va="center", fontsize=7, color="#888")
            else:
                col = "white" if v > 0.6 else "#222"
                # Floor (not round) so 99.86% displays as 99, not 100.
                label = "100" if v >= 0.9999 else f"{int(v * 100)}"
                ax.text(c, r, label, ha="center", va="center", fontsize=6.5, color=col)
    ax.grid(False)
    ax.tick_params(left=False, bottom=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=14)
    cbar.set_label("label present (%)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    out = os.path.join(FIG_DIR, "F17_label_sparsity.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"F17 -> {out}")


def fig_mean_face():
    raw_data_root = os.path.abspath(os.path.join(DATA_ROOT, "..", "Raw Data"))
    specs = [
        ("Small", os.path.join(raw_data_root, "data_small_train.csv"), SMALL_COLS),
        (
            "Large (dedup)",
            os.path.join(DATA_ROOT, "data_large_clean_dedup_train.csv"),
            LARGE_COLS,
        ),
        (
            "Full (dedup)",
            os.path.join(DATA_ROOT, "data_full_clean_dedup_train.csv"),
            SMALL_COLS,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), gridspec_kw={"wspace": 0.08})
    for ax, (name, csv, cols) in zip(axes, specs):
        df = pd.read_csv(csv).sample(n=min(500, len(pd.read_csv(csv))), random_state=0)
        # mean image
        imgs = []
        for s in df["Image"].values:
            imgs.append(np.array(s.split(), dtype=np.float32).reshape(96, 96) / 255.0)
        mean_img = np.mean(imgs, axis=0)
        ax.imshow(mean_img, cmap="gray", vmin=0, vmax=1)
        # scatter individual keypoints + mean
        for c in cols:
            if c.endswith("_x"):
                xc = c
                yc = c[:-2] + "_y"
                xs = df[xc].dropna().values
                ys = df[yc].dropna().values[: len(xs)]
                ax.scatter(
                    xs,
                    ys,
                    s=0.6,
                    color="#D55E00",
                    alpha=0.12,
                    edgecolor="none",
                    zorder=3,
                )
                ax.scatter(
                    [np.nanmean(df[xc])],
                    [np.nanmean(df[yc])],
                    s=22,
                    color="#0072B2",
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=6,
                )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(name.split(" ")[0], fontsize=10, labelpad=6)
    # Legend
    leg = [
        mpatches.Patch(color="#0072B2", label="mean keypoint"),
        mpatches.Patch(color="#D55E00", label="individual keypoints (500 samples)"),
    ]
    fig.legend(
        handles=leg,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8.5,
    )
    out = os.path.join(FIG_DIR, "F18_mean_face.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F18 -> {out}")


def fig_dedup_diagram():
    # Known numbers from PLAN.md / data build report
    sizes_before = {
        "Small": (1284, 428, 428),
        "Large": (4200, 1400, 1400),
        "Full": (4229, 1409, 1411),
    }
    sizes_after = {
        "Small": (1284, 428, 428),  # no pixel dupes
        "Large": (3948, 1360, 1400),  # 287 duplicates removed
        "Full": (3940, 1369, 1411),  # 320 duplicates removed
    }

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.4], wspace=0.25)

    # ---- (a) schematic of the alpha-rule
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")

    # Box helper
    def _box(
        x, y, w, h, text, face="#E6F0F9", edge="#0072B2", fontsize=8.5, bold=False
    ):
        r = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
        )
        ax.add_patch(r)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=("bold" if bold else "normal"),
        )

    def _arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", linewidth=0.8, color="#555"),
        )

    _box(
        0.04,
        0.78,
        0.92,
        0.14,
        "1.  Compute SHA1 hash of every image across (train + val + test)",
        face="#F3F3F3",
        edge="#555",
        bold=True,
    )
    _arrow(0.50, 0.78, 0.50, 0.69)
    _box(
        0.04,
        0.56,
        0.92,
        0.14,
        "2.  Group by SHA1  →  find cross-split duplicates",
        face="#F3F3F3",
        edge="#555",
        bold=True,
    )
    _arrow(0.50, 0.56, 0.50, 0.47)
    _box(
        0.04,
        0.30,
        0.92,
        0.16,
        "3.  Alpha-rule priority:   test  >  val  >  train\n"
        "     (always drop the lower-priority copy)",
        face="#FFF4E6",
        edge="#E69F00",
        bold=True,
        fontsize=9,
    )
    _arrow(0.50, 0.30, 0.50, 0.22)
    _box(
        0.04,
        0.04,
        0.92,
        0.15,
        "4.  Re-save dedup splits; test is preserved bit-for-bit",
        face="#E6F7EE",
        edge="#009E73",
        bold=True,
    )

    # ---- (b) before / after counts
    ax2 = fig.add_subplot(gs[0, 1])
    datasets = ["Small", "Large", "Full"]
    splits = ["train", "val", "test"]
    xs = np.arange(len(datasets))
    w = 0.25
    colors_before = "#BFBFBF"
    colors_after = ["#0072B2", "#009E73", "#D55E00"]  # per split
    # We'll plot "removed" as a hatched delta on top of "after"
    for si, sp in enumerate(splits):
        before = [sizes_before[d][si] for d in datasets]
        after = [sizes_after[d][si] for d in datasets]
        removed = [b - a for a, b in zip(after, before)]
        xoff = xs + (si - 1) * w
        ax2.bar(
            xoff,
            after,
            w,
            color=colors_after[si],
            edgecolor="white",
            linewidth=0.5,
            label=f"{sp}  (kept)",
        )
        if any(r > 0 for r in removed):
            ax2.bar(
                xoff,
                removed,
                w,
                bottom=after,
                color="white",
                edgecolor=colors_after[si],
                linewidth=0.8,
                hatch="////",
                alpha=0.8,
                label=f"{sp}  (removed)" if si == 0 else None,
            )
        for xi, (a, b) in enumerate(zip(after, before)):
            if b - a > 0:
                ax2.text(
                    xs[xi] + (si - 1) * w,
                    b + 40,
                    f"-{b - a}",
                    ha="center",
                    fontsize=7,
                    color=colors_after[si],
                )

    ax2.set_xticks(xs)
    ax2.set_xticklabels(datasets)
    ax2.set_ylabel("Number of samples")
    ax2.legend(loc="upper left", ncol=2, fontsize=7.5)

    out = os.path.join(FIG_DIR, "F19_dedup_diagram.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F19 -> {out}")


def fig_block12_paired(long_df):
    b1 = long_df[long_df["block"] == "01_naive"].copy()
    b2 = long_df[long_df["block"] == "02_dedup"].copy()
    b1_key = b1[b1["dataset"].isin(["large_raw", "full_raw_clean"])].copy()
    b1_key["ds_root"] = b1_key["dataset"].map(
        {"large_raw": "Large", "full_raw_clean": "Full"}
    )
    b2_key = b2.copy()
    b2_key["ds_root"] = b2_key["dataset"].map(
        {"large_dedup": "Large", "full_dedup": "Full"}
    )

    merged = pd.merge(
        b1_key[["ds_root", "model", "best_val_mse"]].rename(
            columns={"best_val_mse": "b1"}
        ),
        b2_key[["ds_root", "model", "best_val_mse"]].rename(
            columns={"best_val_mse": "b2"}
        ),
        on=["ds_root", "model"],
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 3.6), sharey=False, gridspec_kw={"wspace": 0.3}
    )
    for ax, ds in zip(axes, ["Large", "Full"]):
        sub = merged[merged["ds_root"] == ds]
        for _, row in sub.iterrows():
            c = MODEL_COLORS[row["model"]]
            ax.plot(
                [0, 1],
                [row["b1"], row["b2"]],
                marker="o",
                markersize=6,
                color=c,
                linewidth=1.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=MODEL_LABELS[row["model"]],
            )
            pct = (row["b2"] - row["b1"]) / row["b1"] * 100
            ax.annotate(
                f"{pct:+.1f}%", xy=(1.04, row["b2"]), fontsize=7.5, color=c, va="center"
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Block 1\n(raw)", "Block 2\n(dedup)"])
        ax.set_ylabel("Best validation MSE")
        ax.set_xlim(-0.25, 1.45)
    handles, labels = axes[0].get_legend_handles_labels()
    # dedup labels
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="upper right",
        bbox_to_anchor=(0.98, 1.03),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    out = os.path.join(FIG_DIR, "F20_block1_block2_paired.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F20 -> {out}")


def fig_aug_gain_scatter(long_df):
    b3 = long_df[long_df["block"] == "03_aug"].copy()
    # For each (dataset, model): x = S1 MSE, y = best across S2-S5 MSE, colour = model
    rows = []
    for (d, m), g in b3.groupby(["dataset", "model"]):
        s1 = g[g["strategy"] == "S1"]["best_val_mse"]
        others = g[g["strategy"] != "S1"]["best_val_mse"]
        if len(s1) and len(others):
            rows.append(
                {
                    "dataset": d,
                    "model": m,
                    "s1": s1.iloc[0],
                    "best": others.min(),
                    "gain_pct": (others.min() - s1.iloc[0]) / s1.iloc[0] * 100,
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    marker_map = {"small_raw": "o", "large_dedup": "s", "full_dedup": "^"}
    for _, r in df.iterrows():
        c = MODEL_COLORS[r["model"]]
        ax.scatter(
            r["s1"],
            r["best"],
            s=95,
            color=c,
            marker=marker_map[r["dataset"]],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            label=f"{MODEL_LABELS[r['model']]}  ({DATASET_LABELS[r['dataset']]})",
        )
        # annotation
        ax.annotate(
            f"{r['gain_pct']:+.1f}%",
            xy=(r["s1"], r["best"]),
            xytext=(6, 2),
            textcoords="offset points",
            fontsize=7,
            color=c,
        )

    lim = max(df["s1"].max(), df["best"].max()) * 1.1
    ax.plot(
        [0, lim],
        [0, lim],
        color="#888",
        linestyle=(0, (4, 2)),
        linewidth=0.7,
        zorder=1,
        label="y = x  (no gain)",
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("S1 baseline  (val MSE)")
    ax.set_ylabel("Best augmentation result  (val MSE)")
    # build model+dataset legend compactly
    model_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=MODEL_COLORS[m],
            markeredgecolor="white",
            markersize=9,
            label=MODEL_LABELS[m],
        )
        for m in MODELS_ORDER
    ]
    dataset_handles = [
        plt.Line2D(
            [],
            [],
            marker=marker_map[d],
            color="w",
            markerfacecolor="#888",
            markeredgecolor="white",
            markersize=9,
            label=DATASET_LABELS[d],
        )
        for d in ["small_raw", "large_dedup", "full_dedup"]
    ]
    leg1 = ax.legend(
        handles=model_handles,
        loc="upper left",
        title="Model",
        fontsize=7.5,
        title_fontsize=8,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=dataset_handles,
        loc="lower right",
        title="Dataset",
        fontsize=7.5,
        title_fontsize=8,
    )
    out = os.path.join(FIG_DIR, "F21_aug_gain_scatter.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F21 -> {out}")


def fig_rank_trajectory(long_df):
    b3 = long_df[long_df["block"] == "03_aug"].copy()
    strategies = ["S1", "S2", "S3", "S4", "S5"]

    fig, axes = plt.subplots(
        1, 3, figsize=(10.8, 3.4), sharey=True, gridspec_kw={"wspace": 0.2}
    )
    panel_letters = ["a", "b", "c"]
    for k, (ax, d) in enumerate(zip(axes, ["small_raw", "large_dedup", "full_dedup"])):
        sub = b3[b3["dataset"] == d]
        # For each strategy, compute ranks
        rank_rows = []
        for s in strategies:
            g = sub[sub["strategy"] == s].sort_values("best_val_mse")
            for rk, (_, row) in enumerate(g.iterrows(), start=1):
                rank_rows.append({"strategy": s, "model": row["model"], "rank": rk})
        rdf = pd.DataFrame(rank_rows)
        for m in MODELS_ORDER:
            ys = []
            for s in strategies:
                rr = rdf[(rdf["model"] == m) & (rdf["strategy"] == s)]
                ys.append(rr["rank"].iloc[0] if len(rr) else np.nan)
            ax.plot(
                range(len(strategies)),
                ys,
                color=MODEL_COLORS[m],
                marker="o",
                markersize=7,
                linewidth=1.6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=MODEL_LABELS[m] if k == 0 else None,
            )
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["1st", "2nd", "3rd"])
        ax.invert_yaxis()
        if k == 0:
            ax.set_ylabel("Model rank (lower = better)")
        ax.set_xlabel("Strategy")
        panel_label(ax, panel_letters[k])
    axes[0].legend(loc="upper right")
    out = os.path.join(FIG_DIR, "F22_rank_trajectory.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F22 -> {out}")


def fig_sample_scaling(long_df):
    # N_train per dataset (from PLAN.md)
    n_train = {
        "small_raw": 1284,
        "large_dedup": 3948,
        "full_dedup": 3940,
        "large_raw": 4200,
        "full_raw_clean": 4229,
    }
    b3 = long_df[long_df["block"] == "03_aug"].copy()
    # Average best val MSE across S1..S5 per (dataset, model)
    agg = b3.groupby(["dataset", "model"])["best_val_mse"].mean().reset_index()
    agg["n"] = agg["dataset"].map(n_train)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    datasets = ["small_raw", "large_dedup", "full_dedup"]
    for m in MODELS_ORDER:
        sub = agg[agg["model"] == m].sort_values("n")
        ax.plot(
            sub["n"],
            sub["best_val_mse"],
            marker="o",
            markersize=8,
            linewidth=1.6,
            color=MODEL_COLORS[m],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=MODEL_LABELS[m],
        )
        for _, r in sub.iterrows():
            ax.annotate(
                DATASET_LABELS[r["dataset"]].split(" ")[0],
                xy=(r["n"], r["best_val_mse"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=6.5,
                color="#555",
            )
    ax.set_xlabel("Training-set size  (samples)")
    ax.set_ylabel("Mean validation MSE across S1–S5")
    ax.legend(loc="upper right")
    out = os.path.join(FIG_DIR, "F23_sample_size_scaling.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F23 -> {out}")


def _main_build_detail_figures():
    apply_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    long_df, epoch_df = _load()
    fig_label_sparsity()
    fig_mean_face()
    fig_dedup_diagram()
    fig_block12_paired(long_df)
    fig_aug_gain_scatter(long_df)
    fig_rank_trajectory(long_df)
    fig_sample_scaling(long_df)
    print(f"\nDetail figures -> {FIG_DIR}")
