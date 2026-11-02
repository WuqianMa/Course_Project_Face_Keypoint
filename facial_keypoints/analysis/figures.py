"""Analysis / figures for the facial keypoint study."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.style import (
    DATASET_LABELS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS_ORDER,
    STRATEGY_COLORS,
    add_value_labels,
    apply_style,
    panel_label,
)
from ..paths import AGGREGATE_DIR as PAPER_DIR
from ..paths import FIGURE_DIR as FIG_DIR


def _load():
    long_df = pd.read_csv(os.path.join(PAPER_DIR, "tier1_full_long.csv"))
    epoch_df = pd.read_csv(os.path.join(PAPER_DIR, "tier1_full_epoch.csv"))
    return long_df, epoch_df


def fig_block1_bars(long_df):
    # Block 1 (cleaned substrate): data_small has zero cross-split duplicates
    # so it is used as-is from 01_naive; data_large and data_full are taken
    # from 02_dedup (alpha-rule dedup). Three datasets x three models = 9 cells.
    b_small = long_df[
        (long_df["block"] == "01_naive") & (long_df["dataset"] == "small_raw")
    ]
    b_dedup = long_df[long_df["block"] == "02_dedup"]
    b1 = pd.concat([b_small, b_dedup], ignore_index=True)
    datasets = ["small_raw", "large_dedup", "full_dedup"]
    xs = np.arange(len(datasets))
    width = 0.26

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    all_bars = []
    for i, m in enumerate(MODELS_ORDER):
        vals = []
        for d in datasets:
            sub = b1[(b1["dataset"] == d) & (b1["model"] == m)]
            vals.append(sub["best_val_mse"].iloc[0] if len(sub) else np.nan)
        bars = ax.bar(
            xs + (i - 1) * width,
            vals,
            width,
            label=MODEL_LABELS[m],
            color=MODEL_COLORS[m],
            edgecolor="white",
            linewidth=0.8,
        )
        all_bars.append((bars, vals))

    ymax = max(v for _, vs in all_bars for v in vs if not np.isnan(v))
    for bars, _ in all_bars:
        add_value_labels(ax, bars, fmt="{:.3f}", dy=ymax * 0.012, fontsize=6.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([DATASET_LABELS[d].split(" ")[0] for d in datasets])
    ax.set_ylabel("Best validation MSE")
    ax.set_ylim(0, ymax * 1.15)
    ax.legend(loc="upper right", ncol=1)
    out = os.path.join(FIG_DIR, "F1_block1_bars.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F1 -> {out}")


def fig_leakage_delta(long_df):
    b1 = long_df[long_df["block"] == "01_naive"].copy()
    b2 = long_df[long_df["block"] == "02_dedup"].copy()
    b1_key = b1[b1["dataset"].isin(["large_raw", "full_raw_clean"])].copy()
    b1_key["ds_root"] = b1_key["dataset"].map(
        {"large_raw": "data_large", "full_raw_clean": "data_full"}
    )
    b2_key = b2.copy()
    b2_key["ds_root"] = b2_key["dataset"].map(
        {"large_dedup": "data_large", "full_dedup": "data_full"}
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
    merged["delta_pct"] = (merged["b2"] - merged["b1"]) / merged["b1"] * 100

    datasets = ["data_large", "data_full"]
    ds_pretty = {
        "data_large": "Large (4 200 / 1 400 / 1 400)",
        "data_full": "Full (4 229 / 1 409 / 1 411)",
    }
    xs = np.arange(len(datasets))
    width = 0.26

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    all_bars = []
    for i, m in enumerate(MODELS_ORDER):
        vals = []
        for d in datasets:
            sub = merged[(merged["ds_root"] == d) & (merged["model"] == m)]
            vals.append(sub["delta_pct"].iloc[0] if len(sub) else np.nan)
        bars = ax.bar(
            xs + (i - 1) * width,
            vals,
            width,
            label=MODEL_LABELS[m],
            color=MODEL_COLORS[m],
            edgecolor="white",
            linewidth=0.8,
        )
        all_bars.append((bars, vals))

    vmax = max(abs(v) for _, vs in all_bars for v in vs if not np.isnan(v))
    for bars, _ in all_bars:
        add_value_labels(ax, bars, fmt="{:+.1f}%", dy=vmax * 0.04, fontsize=6.5)

    ax.axhline(0, color="#333", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([ds_pretty[d] for d in datasets])
    ax.set_ylabel(r"$\Delta$ val MSE after dedup (%)")
    ax.set_ylim(-vmax * 1.3, vmax * 1.5)
    ax.legend(loc="upper left")
    out = os.path.join(FIG_DIR, "F2_leakage_delta.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F2 -> {out}")


def fig_block3_factorial(long_df):
    b3 = long_df[long_df["block"] == "03_aug"].copy()
    datasets = ["small_raw", "large_dedup", "full_dedup"]
    strategies = sorted(b3["strategy"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=False)
    panel_letters = ["a", "b", "c"]
    for k, (ax, d) in enumerate(zip(axes, datasets)):
        xs = np.arange(len(strategies))
        width = 0.26
        for i, m in enumerate(MODELS_ORDER):
            vals = []
            for s in strategies:
                sub = b3[
                    (b3["dataset"] == d) & (b3["model"] == m) & (b3["strategy"] == s)
                ]
                vals.append(sub["best_val_mse"].iloc[0] if len(sub) else np.nan)
            ax.bar(
                xs + (i - 1) * width,
                vals,
                width,
                label=MODEL_LABELS[m] if k == 0 else None,
                color=MODEL_COLORS[m],
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(strategies)
        ax.set_xlabel("Augmentation strategy")
        if k == 0:
            ax.set_ylabel("Best validation MSE")
        panel_label(ax, panel_letters[k])
    axes[0].legend(loc="upper right", ncol=1)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "F3_block3_factorial.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F3 -> {out}")


def fig_learning_curves(long_df, epoch_df):
    b3_e = epoch_df[epoch_df["block"] == "03_aug"].copy()
    b3_s = long_df[long_df["block"] == "03_aug"].copy()

    for dset in ["small_raw", "large_dedup", "full_dedup"]:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=False)
        panel_letters = ["a", "b", "c"]
        for k, (ax, m) in enumerate(zip(axes, MODELS_ORDER)):
            sub_s = b3_s[(b3_s["dataset"] == dset) & (b3_s["model"] == m)]
            best_s = sub_s.loc[sub_s["best_val_mse"].idxmin(), "strategy"]
            for s, style in [
                (
                    "S1",
                    {
                        "color": STRATEGY_COLORS["S1"],
                        "linestyle": (0, (4, 2)),
                        "label": "S1 (photometric only)",
                    },
                ),
                (
                    best_s,
                    {
                        "color": STRATEGY_COLORS.get(best_s, "#000"),
                        "linestyle": "-",
                        "label": f"{best_s} (best)",
                    },
                ),
            ]:
                ep = b3_e[
                    (b3_e["dataset"] == dset)
                    & (b3_e["model"] == m)
                    & (b3_e["strategy"] == s)
                ].sort_values("epoch")
                ax.plot(ep["epoch"], ep["val_mse"], linewidth=1.4, **style)
                # mark best epoch
                if s == best_s:
                    be = sub_s[sub_s["strategy"] == best_s]["best_epoch"].iloc[0]
                    bv = sub_s[sub_s["strategy"] == best_s]["best_val_mse"].iloc[0]
                    ax.scatter(
                        [be],
                        [bv],
                        s=22,
                        color=STRATEGY_COLORS.get(best_s, "#000"),
                        edgecolor="white",
                        linewidth=0.8,
                        zorder=5,
                    )
            ax.set_xlabel("Epoch")
            if k == 0:
                ax.set_ylabel("Validation MSE")
            ax.legend(loc="upper right")
            panel_label(ax, panel_letters[k])
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"F4_learning_curves_{dset}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"F4 ({dset}) -> {out}")


def fig_heatmap(long_df):
    b3 = long_df[long_df["block"] == "03_aug"].copy()
    strategies = sorted(b3["strategy"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 2.9), gridspec_kw={"wspace": 0.10})
    fig.subplots_adjust(left=0.10, right=0.90)
    panel_letters = ["a", "b", "c"]
    last_im = None
    for k, (ax, dset) in enumerate(
        zip(axes, ["small_raw", "large_dedup", "full_dedup"])
    ):
        mat = np.zeros((len(MODELS_ORDER), len(strategies)))
        for i, m in enumerate(MODELS_ORDER):
            s1_sub = b3[
                (b3["dataset"] == dset) & (b3["model"] == m) & (b3["strategy"] == "S1")
            ]
            s1 = s1_sub["best_val_mse"].iloc[0] if len(s1_sub) else np.nan
            for j, s in enumerate(strategies):
                sub = b3[
                    (b3["dataset"] == dset) & (b3["model"] == m) & (b3["strategy"] == s)
                ]
                v = sub["best_val_mse"].iloc[0] if len(sub) else np.nan
                mat[i, j] = (s1 - v) / s1 * 100 if s1 and not np.isnan(s1) else np.nan
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-40, vmax=40)
        last_im = im
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies)
        ax.set_yticks(range(len(MODELS_ORDER)))
        if k == 0:
            ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS_ORDER], fontsize=9)
            ax.tick_params(axis="y", left=False, labelleft=True, pad=4)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False, labelleft=False)
        for i in range(len(MODELS_ORDER)):
            for j in range(len(strategies)):
                v = mat[i, j]
                if not np.isnan(v):
                    color = "white" if abs(v) > 22 else "#111"
                    ax.text(
                        j,
                        i,
                        f"{v:+.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                        fontweight="bold",
                    )
        ax.grid(False)
        ax.tick_params(axis="x", bottom=False)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_title(
            f"({panel_letters[k]}) {DATASET_LABELS[dset].split(' ')[0]}",
            fontsize=9,
            pad=6,
        )

    # shared colorbar
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.02, aspect=18)
    cbar.set_label("Improvement vs S1 (%)  (positive = better)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.4)

    out = os.path.join(FIG_DIR, "F5_strategy_heatmap.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"F5 -> {out}")


def _main_build_figures():
    apply_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    long_df, epoch_df = _load()
    fig_block1_bars(long_df)
    fig_leakage_delta(long_df)
    fig_block3_factorial(long_df)
    fig_learning_curves(long_df, epoch_df)
    fig_heatmap(long_df)
    print(f"\nAll figures -> {FIG_DIR}")
