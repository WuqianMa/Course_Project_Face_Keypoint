"""Analysis / deep for the facial keypoint study."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt

from ..analysis.style import (
    DATASET_COLORS,
    DATASET_LABELS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS_ORDER,
    add_value_labels,
    apply_style,
    panel_label,
)
from ..paths import AGGREGATE_DIR as PAPER_DIR
from ..paths import FIGURE_DIR as FIG_DIR
from ..paths import TABLE_DIR as TAB_DIR


def _load():
    long_df = pd.read_csv(os.path.join(PAPER_DIR, "tier1_full_long.csv"))
    epoch_df = pd.read_csv(os.path.join(PAPER_DIR, "tier1_full_epoch.csv"))
    return long_df, epoch_df


def _simple_docx(title, caption, df, out_path, nd=5, footer=None):
    doc = Document()
    styles = doc.styles["Normal"]
    styles.font.name = "Times New Roman"
    styles.font.size = Pt(10)

    h = doc.add_paragraph()
    run = h.add_run(title)
    run.bold = True
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"

    c = doc.add_paragraph(caption)
    for r in c.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)
        r.italic = True

    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid"
    hdr = t.rows[0].cells
    for j, col in enumerate(df.columns):
        p = hdr[j].paragraphs[0]
        r = p.add_run(str(col))
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)

    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, val in enumerate(row.tolist()):
            if isinstance(val, float) and not np.isnan(val):
                val_str = f"{val:.{nd}g}" if abs(val) < 1 else f"{val:.3f}"
            elif pd.isna(val):
                val_str = ""
            else:
                val_str = str(val)
            p = cells[j].paragraphs[0]
            r = p.add_run(val_str)
            r.font.name = "Times New Roman"
            r.font.size = Pt(9)

    if footer:
        f = doc.add_paragraph(footer)
        for r in f.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(8)
            r.italic = True

    doc.save(out_path)


def build_T6(long_df, epoch_df):
    rows = []
    for _, s in long_df.iterrows():
        sub = epoch_df[
            (epoch_df["dataset"] == s["dataset"])
            & (epoch_df["model"] == s["model"])
            & (epoch_df["strategy"] == s["strategy"])
            & (epoch_df["epoch"] == s["best_epoch"])
        ]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "block": s["block"],
                "dataset": s["dataset"],
                "model": s["model"],
                "strategy": s["strategy"],
                "best_epoch": int(s["best_epoch"])
                if not pd.isna(s["best_epoch"])
                else None,
                "train_mse_at_best": r["train_mse"],
                "val_mse_at_best": r["val_mse"],
                "gap": r["val_mse"] - r["train_mse"],
                "gap_ratio": r["val_mse"] / r["train_mse"]
                if r["train_mse"] > 0
                else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    csv_p = os.path.join(TAB_DIR, "T6_overfit_gap.csv")
    docx_p = os.path.join(TAB_DIR, "T6_overfit_gap.docx")
    df.to_csv(csv_p, index=False)

    summary = (
        df.groupby(["block", "model"])
        .agg(
            mean_gap=("gap", "mean"),
            mean_ratio=("gap_ratio", "mean"),
            n=("gap", "size"),
        )
        .reset_index()
    )
    _simple_docx(
        "Table T6: Train-val gap at best epoch (summary by block x model)",
        "Mean (val MSE - train MSE) and mean ratio across all cells in each "
        "(block, model) group. Large positive gap = overfitting; ratio near 1 "
        "= well-generalised.",
        summary,
        docx_p,
        nd=4,
        footer="Gap computed at the epoch that minimised val MSE. "
        "Ratio = val/train at that epoch.",
    )
    print(f"T6 -> {csv_p}, {docx_p}")
    return df


def build_T7(long_df):
    df = long_df.copy()
    df["best_epoch"] = df["best_epoch"].astype("Int64")
    summary = (
        df.groupby(["block", "model"])
        .agg(
            mean_best_epoch=("best_epoch", "mean"),
            median_best_epoch=("best_epoch", "median"),
            min_best_epoch=("best_epoch", "min"),
            max_best_epoch=("best_epoch", "max"),
            n=("best_epoch", "size"),
        )
        .reset_index()
    )
    csv_p = os.path.join(TAB_DIR, "T7_best_epoch.csv")
    summary.to_csv(csv_p, index=False)
    print(f"T7 -> {csv_p}")
    return df


def build_T8(long_df):
    b3 = long_df[long_df["block"] == "03_aug"].copy()
    strategies = ["S1", "S2", "S3", "S4", "S5"]
    rows = []
    for (d, m), g in b3.groupby(["dataset", "model"]):
        vals = {
            s: g[g["strategy"] == s]["best_val_mse"].iloc[0]
            if len(g[g["strategy"] == s])
            else np.nan
            for s in strategies
        }
        s1 = vals["S1"]
        row = {"dataset": d, "model": m, "S1": s1}
        for prev, cur in zip(strategies[:-1], strategies[1:]):
            if pd.notna(vals[cur]) and pd.notna(vals[prev]) and s1 > 0:
                row[f"{prev}->{cur}_delta"] = vals[cur] - vals[prev]
                row[f"{prev}->{cur}_pct"] = (vals[cur] - vals[prev]) / s1 * 100
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_p = os.path.join(TAB_DIR, "T8_marginal_aug.csv")
    df.to_csv(csv_p, index=False)
    print(f"T8 -> {csv_p}")
    return df


def build_T9(long_df, epoch_df):
    df = long_df.copy()
    tot_sec = (
        epoch_df.groupby(["dataset", "model", "strategy"])["sec"].sum().reset_index()
    )
    tot_sec = tot_sec.rename(columns={"sec": "total_train_sec"})
    df = df.merge(tot_sec, on=["dataset", "model", "strategy"], how="left")
    df["sec_per_epoch"] = df["total_train_sec"] / 30.0
    worst_by_block = df.groupby("block")["best_val_mse"].transform("max")
    df["mse_improv_over_worst"] = worst_by_block - df["best_val_mse"]
    df["mse_gain_per_sec"] = df["mse_improv_over_worst"] / df["total_train_sec"]
    out = df[
        [
            "block",
            "dataset",
            "model",
            "strategy",
            "best_val_mse",
            "total_train_sec",
            "sec_per_epoch",
            "mse_gain_per_sec",
        ]
    ]
    csv_p = os.path.join(TAB_DIR, "T9_training_efficiency.csv")
    out.to_csv(csv_p, index=False)
    print(f"T9 -> {csv_p}")
    return df


def build_T10(long_df):
    b3 = long_df[long_df["block"] == "03_aug"].copy()
    rows = []
    for (d, s), g in b3.groupby(["dataset", "strategy"]):
        g2 = g.sort_values("best_val_mse").reset_index(drop=True)
        for rk, (_, r) in enumerate(g2.iterrows(), start=1):
            rows.append(
                {
                    "dataset": d,
                    "strategy": s,
                    "model": r["model"],
                    "rank": rk,
                    "val_mse": r["best_val_mse"],
                }
            )
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index=["dataset", "model"], columns="strategy", values="rank"
    ).reset_index()
    pivot["rank_changed"] = pivot[["S1", "S2", "S3", "S4", "S5"]].apply(
        lambda r: r.nunique() > 1, axis=1
    )
    csv_p = os.path.join(TAB_DIR, "T10_ranking_stability.csv")
    pivot.to_csv(csv_p, index=False)
    print(f"T10 -> {csv_p}")
    return pivot


def fig_overfit_gap(T6_df):
    b3 = T6_df[T6_df["block"] == "03_aug"].copy()
    datasets = ["small_raw", "large_dedup", "full_dedup"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    panel_letters = ["a", "b", "c"]
    for k, (ax, d) in enumerate(zip(axes, datasets)):
        sub = b3[b3["dataset"] == d]
        strategies = sorted(sub["strategy"].unique())
        xs = np.arange(len(strategies))
        width = 0.26
        for i, m in enumerate(MODELS_ORDER):
            vals = []
            for s in strategies:
                row = sub[(sub["model"] == m) & (sub["strategy"] == s)]
                vals.append(row["gap"].iloc[0] if len(row) else np.nan)
            ax.bar(
                xs + (i - 1) * width,
                vals,
                width,
                label=MODEL_LABELS[m] if k == 0 else None,
                color=MODEL_COLORS[m],
                edgecolor="white",
                linewidth=0.6,
            )
        ax.axhline(0, color="#333", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(strategies)
        ax.set_xlabel("Strategy")
        if k == 0:
            ax.set_ylabel("Val MSE − Train MSE")
        panel_label(ax, panel_letters[k])
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "F6_overfit_gap.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F6 -> {out}")


def fig_best_epoch(long_df):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bins = np.arange(1, 32, 2)
    data = [
        long_df[long_df["model"] == m]["best_epoch"].dropna().values
        for m in MODELS_ORDER
    ]
    colors = [MODEL_COLORS[m] for m in MODELS_ORDER]
    labels = [MODEL_LABELS[m] for m in MODELS_ORDER]
    ax.hist(
        data,
        bins=bins,
        stacked=True,
        color=colors,
        label=labels,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Epoch of best validation MSE")
    ax.set_ylabel("Number of runs")
    ax.legend(loc="upper left")
    out = os.path.join(FIG_DIR, "F7_best_epoch_hist.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F7 -> {out}")


def fig_marginal_aug(T8_df):
    steps = ["S1->S2", "S2->S3", "S3->S4", "S4->S5"]
    step_labels = ["S1→S2", "S2→S3", "S3→S4", "S4→S5"]
    datasets = ["small_raw", "large_dedup", "full_dedup"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    panel_letters = ["a", "b", "c"]
    for k, (ax, m) in enumerate(zip(axes, MODELS_ORDER)):
        sub = T8_df[T8_df["model"] == m]
        xs = np.arange(len(steps))
        width = 0.26
        for i, d in enumerate(datasets):
            row = sub[sub["dataset"] == d]
            if len(row) == 0:
                continue
            vals = [
                row[f"{s}_pct"].iloc[0] if f"{s}_pct" in row.columns else np.nan
                for s in steps
            ]
            ax.bar(
                xs + (i - 1) * width,
                vals,
                width,
                label=DATASET_LABELS[d] if k == 0 else None,
                color=DATASET_COLORS[d],
                edgecolor="white",
                linewidth=0.6,
            )
        ax.axhline(0, color="#333", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(step_labels, rotation=0)
        if k == 0:
            ax.set_ylabel("Marginal Δ (% of S1)")
        panel_label(ax, panel_letters[k])
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "F8_marginal_aug.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F8 -> {out}")


def fig_convergence(long_df, epoch_df):
    rows = []
    for _, s in long_df.iterrows():
        ep = epoch_df[
            (epoch_df["dataset"] == s["dataset"])
            & (epoch_df["model"] == s["model"])
            & (epoch_df["strategy"] == s["strategy"])
        ].sort_values("epoch")
        target = 1.5 * s["best_val_mse"]
        hit = ep[ep["val_mse"] <= target]
        if len(hit):
            rows.append(
                {
                    "block": s["block"],
                    "dataset": s["dataset"],
                    "model": s["model"],
                    "strategy": s["strategy"],
                    "epoch_to_1p5x": int(hit.iloc[0]["epoch"]),
                }
            )
    df = pd.DataFrame(rows)
    csv_p = os.path.join(TAB_DIR, "T11_convergence_speed.csv")
    df.to_csv(csv_p, index=False)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    data = [df[df["model"] == m]["epoch_to_1p5x"].values for m in MODELS_ORDER]
    positions = np.arange(len(MODELS_ORDER)) + 1
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="#333", linewidth=0.7),
        capprops=dict(color="#333", linewidth=0.7),
        flierprops=dict(
            marker="o", markerfacecolor="none", markeredgecolor="#666", markersize=3.5
        ),
    )
    for patch, m in zip(bp["boxes"], MODELS_ORDER):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.45)
        patch.set_edgecolor(MODEL_COLORS[m])
        patch.set_linewidth(0.8)

    # overlay individual points with jitter
    rng = np.random.default_rng(0)
    for i, (m, d) in enumerate(zip(MODELS_ORDER, data)):
        if len(d):
            jitter = rng.normal(0, 0.06, size=len(d))
            ax.scatter(
                np.full_like(d, positions[i], dtype=float) + jitter,
                d,
                color=MODEL_COLORS[m],
                s=14,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS_ORDER])
    ax.set_ylabel("Epochs to reach 1.5× best-val MSE")
    out = os.path.join(FIG_DIR, "F9_convergence_speed.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F9 -> {out}, T11 -> {csv_p}")


def fig_scatter(long_df):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for m in MODELS_ORDER:
        sub = long_df[long_df["model"] == m]
        ax.scatter(
            sub["final_train_mse"],
            sub["final_val_mse"],
            label=MODEL_LABELS[m],
            color=MODEL_COLORS[m],
            edgecolor="white",
            s=42,
            alpha=0.85,
            linewidth=0.6,
        )
    lims = [
        0,
        max(long_df["final_train_mse"].max(), long_df["final_val_mse"].max()) * 1.05,
    ]
    ax.plot(
        lims, lims, color="#555", linestyle=(0, (4, 2)), linewidth=0.8, label="y = x"
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Final train MSE")
    ax.set_ylabel("Final validation MSE")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    # annotate quadrant
    ax.text(
        lims[1] * 0.04,
        lims[1] * 0.96,
        "above y = x\n→ overfit",
        fontsize=7.5,
        color="#555",
        va="top",
        ha="left",
        style="italic",
    )
    out = os.path.join(FIG_DIR, "F10_train_val_scatter.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F10 -> {out}")


def fig_aug_sensitivity(long_df):
    b3 = long_df[long_df["block"] == "03_aug"].copy()
    aug_only = b3[b3["strategy"] != "S1"]
    agg = (
        aug_only.groupby(["dataset", "model"])["best_val_mse"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    agg["range"] = agg["max"] - agg["min"]
    agg["cv"] = agg["std"] / agg["mean"] * 100
    csv_p = os.path.join(TAB_DIR, "T12_aug_sensitivity.csv")
    agg.to_csv(csv_p, index=False)

    datasets = ["small_raw", "large_dedup", "full_dedup"]
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    xs = np.arange(len(datasets))
    width = 0.26
    all_bars = []
    for i, m in enumerate(MODELS_ORDER):
        vals = []
        for d in datasets:
            row = agg[(agg["dataset"] == d) & (agg["model"] == m)]
            vals.append(row["cv"].iloc[0] if len(row) else np.nan)
        bars = ax.bar(
            xs + (i - 1) * width,
            vals,
            width,
            label=MODEL_LABELS[m],
            color=MODEL_COLORS[m],
            edgecolor="white",
            linewidth=0.6,
        )
        all_bars.append((bars, vals))
    vmax = max(v for _, vs in all_bars for v in vs if not np.isnan(v))
    for bars, _ in all_bars:
        add_value_labels(ax, bars, fmt="{:.1f}", dy=vmax * 0.02, fontsize=6.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("CV of best val MSE across S2–S5 (%)")
    ax.set_ylim(0, vmax * 1.18)
    ax.legend(loc="upper right")
    out = os.path.join(FIG_DIR, "F11_aug_sensitivity.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"F11 -> {out}, T12 -> {csv_p}")


def _main_build_deep_analysis():
    apply_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    long_df, epoch_df = _load()

    T6 = build_T6(long_df, epoch_df)
    build_T7(long_df)
    T8 = build_T8(long_df)
    build_T9(long_df, epoch_df)
    build_T10(long_df)

    fig_overfit_gap(T6)
    fig_best_epoch(long_df)
    fig_marginal_aug(T8)
    fig_convergence(long_df, epoch_df)
    fig_scatter(long_df)
    fig_aug_sensitivity(long_df)
    print(f"\nDeep-analysis outputs -> {TAB_DIR} and {FIG_DIR}")
