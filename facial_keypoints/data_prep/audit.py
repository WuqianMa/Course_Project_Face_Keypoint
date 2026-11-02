"""Data prep / audit for the facial keypoint study."""

import hashlib
import json
import os
from collections import Counter
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

matplotlib.use("Agg")
from ..paths import CLEAN_DIR as CLEAN
from ..paths import DATA_DIR as DATA
from ..paths import EDA_DIR as OUT
from ..paths import PROJECT_ROOT as SVROOT

OUT_T = os.path.join(OUT, "tables")
OUT_F = os.path.join(OUT, "figures")


DATASETS = {
    "data_small": {
        "train": os.path.join(DATA, "data_small_train.csv"),
        "val": os.path.join(DATA, "data_small_val.csv"),
        "test": os.path.join(DATA, "data_small_test.csv"),
    },
    "data_large": {
        "train": os.path.join(DATA, "data_large_train.csv"),
        "val": os.path.join(DATA, "data_large_val.csv"),
        "test": os.path.join(DATA, "data_large_test.csv"),
    },
    "data_full": {
        "train": os.path.join(CLEAN, "data_full_clean_train.csv"),
        "val": os.path.join(CLEAN, "data_full_clean_val.csv"),
        "test": os.path.join(CLEAN, "data_full_clean_test.csv"),
    },
}

SPLIT_ORDER = ["train", "val", "test"]

rcParams["font.family"] = "Times New Roman"

rcParams["axes.grid"] = True

rcParams["grid.alpha"] = 0.25

rcParams["grid.linestyle"] = "--"

rcParams["savefig.bbox"] = "tight"


def sha1_series(s: pd.Series) -> pd.Series:
    """SHA1 of each Image pixel string."""
    return s.astype(str).map(lambda x: hashlib.sha1(x.encode()).hexdigest())


def load_dataset(name: str) -> dict:
    """Load three splits of a dataset, return dict of DataFrames with sha1 column."""
    out = {}
    for split in SPLIT_ORDER:
        path = DATASETS[name][split]
        df = pd.read_csv(path, usecols=["Image"])
        df["__sha1__"] = sha1_series(df["Image"])
        df["__split__"] = split
        df = df.drop(columns=["Image"])  # free memory
        out[split] = df
    return out


def audit_dataset(name: str, splits: dict) -> dict:
    """Compute duplicate stats for one dataset."""
    concat = pd.concat([splits[s] for s in SPLIT_ORDER], ignore_index=True)
    n_rows = len(concat)
    n_unique = int(concat["__sha1__"].nunique())

    # Within-raw: row_count - unique_count
    within_raw_dup_rows = int(n_rows - n_unique)

    # Duplicate group sizes (for rows that belong to any group of size >= 2)
    vc = concat["__sha1__"].value_counts()
    dup_groups = vc[vc >= 2]
    group_size_dist = Counter(dup_groups.tolist())

    # Within-split
    within_split = {}
    for s in SPLIT_ORDER:
        part_vc = splits[s]["__sha1__"].value_counts()
        within_split[s] = {
            "n_rows": int(len(splits[s])),
            "n_unique": int(part_vc.size),
            "dup_rows": int(int(len(splits[s])) - int(part_vc.size)),
            "dup_groups": int((part_vc >= 2).sum()),
        }

    # Cross-split: for each SHA1, which splits does it appear in?
    split_sets = {s: set(splits[s]["__sha1__"]) for s in SPLIT_ORDER}
    cross = {}
    for a in SPLIT_ORDER:
        for b in SPLIT_ORDER:
            if a < b:
                cross[f"{a}_x_{b}"] = int(len(split_sets[a] & split_sets[b]))
    # Triple overlap
    cross["all_three"] = int(
        len(split_sets["train"] & split_sets["val"] & split_sets["test"])
    )

    return {
        "dataset": name,
        "n_rows_total": n_rows,
        "n_unique_sha1": n_unique,
        "within_raw_dup_rows": within_raw_dup_rows,
        "frac_dup_rows": within_raw_dup_rows / n_rows if n_rows else 0.0,
        "n_dup_groups": int(len(dup_groups)),
        "max_group_size": int(dup_groups.max()) if len(dup_groups) else 1,
        "group_size_distribution": dict(sorted(group_size_dist.items())),
        "within_split": within_split,
        "cross_split": cross,
    }


def _eda_write_tables(audits: list):
    # 01 — per-dataset summary (long)
    rows = []
    for a in audits:
        rows.append(
            {
                "dataset": a["dataset"],
                "n_rows_total": a["n_rows_total"],
                "n_unique_sha1": a["n_unique_sha1"],
                "within_raw_dup_rows": a["within_raw_dup_rows"],
                "frac_dup_rows": round(a["frac_dup_rows"], 4),
                "n_dup_groups": a["n_dup_groups"],
                "max_group_size": a["max_group_size"],
            }
        )
        for s in SPLIT_ORDER:
            ws = a["within_split"][s]
            rows.append(
                {
                    "dataset": f"{a['dataset']} [{s}]",
                    "n_rows_total": ws["n_rows"],
                    "n_unique_sha1": ws["n_unique"],
                    "within_raw_dup_rows": ws["dup_rows"],
                    "frac_dup_rows": round(ws["dup_rows"] / ws["n_rows"], 4)
                    if ws["n_rows"]
                    else 0.0,
                    "n_dup_groups": ws["dup_groups"],
                    "max_group_size": "",
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(OUT_T, "01_dup_counts.csv"), index=False)

    # 02 — cross-split
    rows = []
    for a in audits:
        row = {"dataset": a["dataset"]}
        row.update(a["cross_split"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_T, "02_dup_cross_split.csv"), index=False
    )

    # 03 — group size distribution
    rows = []
    for a in audits:
        for size, cnt in a["group_size_distribution"].items():
            rows.append(
                {
                    "dataset": a["dataset"],
                    "group_size": int(size),
                    "n_groups": int(cnt),
                }
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_T, "03_dup_group_sizes.csv"), index=False
    )


def _eda_write_figures(audits: list):
    ds_names = [a["dataset"] for a in audits]

    # --- 01: bars of within-raw duplicate rows per dataset ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ys = [a["within_raw_dup_rows"] for a in audits]
    xs = np.arange(len(ds_names))
    ax.bar(xs, ys, width=0.55, color="0.35", edgecolor="black", linewidth=0.7)
    for i, v in enumerate(ys):
        ax.text(xs[i], v, f"{v}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(ds_names, fontsize=10)
    ax.set_ylabel("Duplicate rows (within-raw)")
    ax.set_title(
        "Pixel-exact duplicate rows per dataset (concatenated splits)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_F, "01_dup_counts_bars.pdf"))
    plt.close(fig)

    # --- 02: cross-split overlap 3x3 heatmaps ------------------------------
    fig, axes = plt.subplots(
        1, len(audits), figsize=(4.5 * len(audits), 4.2), squeeze=False
    )
    axes = axes[0]
    for ax, a in zip(axes, audits):
        mat = np.zeros((3, 3), dtype=int)
        sets = {}
        for s in SPLIT_ORDER:
            sets[s] = None  # placeholder; we use cross_split dict instead
        for i, si in enumerate(SPLIT_ORDER):
            for j, sj in enumerate(SPLIT_ORDER):
                if i == j:
                    mat[i, j] = a["within_split"][si]["n_unique"]
                elif si < sj:
                    mat[i, j] = a["cross_split"][f"{si}_x_{sj}"]
                else:
                    mat[i, j] = a["cross_split"][f"{sj}_x_{si}"]
        im = ax.imshow(mat, cmap="Greys", aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(mat[i, j]),
                    ha="center",
                    va="center",
                    color="red" if mat[i, j] > 0 and i != j else "black",
                    fontsize=10,
                )
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(SPLIT_ORDER, fontsize=9)
        ax.set_yticklabels(SPLIT_ORDER, fontsize=9)
        ax.set_title(
            f"{a['dataset']}\n(diag: unique-sha1; off-diag: overlap)", fontsize=10
        )
    fig.suptitle("Cross-split pixel duplication by dataset", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_F, "02_cross_split_heat.pdf"))
    plt.close(fig)

    # --- 03: histogram of duplicate group sizes (per-dataset, overlaid) ----
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    linestyles = ["-", "--", ":"]
    shades = ["0.2", "0.45", "0.7"]
    all_sizes = set()
    for a in audits:
        all_sizes.update(int(k) for k in a["group_size_distribution"].keys())
    max_size = max(all_sizes) if all_sizes else 2
    bins = np.arange(1.5, max_size + 1.5, 1)
    for i, a in enumerate(audits):
        sizes = []
        for k, cnt in a["group_size_distribution"].items():
            sizes.extend([int(k)] * int(cnt))
        if not sizes:
            continue
        ax.hist(
            sizes,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=shades[i % len(shades)],
            linestyle=linestyles[i % len(linestyles)],
            label=a["dataset"],
        )
    ax.set_xlabel("Group size (# of rows sharing the same SHA1)")
    ax.set_ylabel("Number of duplicate groups")
    ax.set_title("Distribution of duplicate-group sizes", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_F, "03_dup_group_size_hist.pdf"))
    plt.close(fig)


def _main_eda_dup():
    os.makedirs(OUT_T, exist_ok=True)
    os.makedirs(OUT_F, exist_ok=True)
    t0 = datetime.now()
    print(f"[start] {t0}")
    audits = []
    for name in DATASETS:
        print(f"[load ] {name}")
        splits = load_dataset(name)
        for s in SPLIT_ORDER:
            print(f"        {s}: {len(splits[s])} rows")
        a = audit_dataset(name, splits)
        print(
            f"[audit] {name}: within-raw dup rows = {a['within_raw_dup_rows']} "
            f"({a['frac_dup_rows']:.4f}), n_dup_groups = {a['n_dup_groups']}, "
            f"max_group_size = {a['max_group_size']}"
        )
        for s in SPLIT_ORDER:
            ws = a["within_split"][s]
            print(
                f"        within-{s}: {ws['dup_rows']} dup rows in {ws['dup_groups']} groups"
            )
        for k, v in a["cross_split"].items():
            if v > 0:
                print(f"        CROSS {k}: {v} overlapping SHA1")
        audits.append(a)

    print(f"[write] tables -> {OUT_T}")
    _eda_write_tables(audits)
    print(f"[write] figures -> {OUT_F}")
    _eda_write_figures(audits)

    report_path = os.path.join(OUT, "eda_image_duplicates_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "generated": str(t0),
                "sv_root": SVROOT,
                "datasets": audits,
            },
            f,
            indent=2,
        )
    print(f"[ok   ] report -> {report_path}")

    t1 = datetime.now()
    print(f"[done ] elapsed: {t1 - t0}")
