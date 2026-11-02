"""Analysis / tables for the facial keypoint study."""

import os

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from ..analysis.style import apply_style
from ..paths import AGGREGATE_DIR as PAPER_DIR
from ..paths import TABLE_DIR as TABLES_DIR

DATASETS_ORDER = [
    "small_raw",
    "large_raw",
    "full_raw_clean",
    "large_dedup",
    "full_dedup",
]

MODELS_ORDER = ["mlp2", "cnn_shallow", "resnet18_pretrained"]

STRATEGY_ORDER = ["none", "S1", "S2", "S3", "S4", "S5"]


def _apply_times(doc):
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10)


def _fmt(v, nd=5):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _build_docx(title, caption, df, out_path, nd=5, footer=None):
    doc = Document()
    _apply_times(doc)

    ttl = doc.add_paragraph()
    ttl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = ttl.add_run(title)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)

    cap = doc.add_paragraph()
    cap_run = cap.add_run(caption)
    cap_run.italic = True
    cap_run.font.name = "Times New Roman"
    cap_run.font.size = Pt(9)

    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Light Grid"
    hdr = table.rows[0].cells
    for i, col in enumerate(df.columns):
        p = hdr[i].paragraphs[0]
        r = p.add_run(str(col))
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, col in enumerate(df.columns):
            p = cells[i].paragraphs[0]
            r = p.add_run(_fmt(row[col], nd=nd))
            r.font.name = "Times New Roman"
            r.font.size = Pt(9)

    if footer:
        note = doc.add_paragraph()
        nr = note.add_run(footer)
        nr.italic = True
        nr.font.name = "Times New Roman"
        nr.font.size = Pt(8)

    doc.save(out_path)


def build_T2(long_df):
    b1 = long_df[long_df["block"] == "01_naive"].copy()
    wide = b1.pivot_table(
        index="dataset", columns="model", values="best_val_mse", aggfunc="first"
    ).reindex(index=["small_raw", "large_raw", "full_raw_clean"])[MODELS_ORDER]
    wide.columns.name = None
    wide = wide.reset_index().rename(columns={"dataset": "Dataset"})

    csv_path = os.path.join(TABLES_DIR, "T2_block1_baselines.csv")
    docx_path = os.path.join(TABLES_DIR, "T2_block1_baselines.docx")
    wide.to_csv(csv_path, index=False)

    caption = (
        "Table T2. Block 1 baselines (S1/strategy=none). Best validation MSE "
        "across 30 training epochs, per dataset and model. data_full uses "
        "the non-dedup clean split trained with masked MSE; "
        "small/large use naive MSE."
    )
    footer = (
        "Note: seeds=0; 30 epochs; batch size 128. "
        "n_train (small/large/full) = 1284 / 4200 / 4229. "
        "Best val MSE uses 1.0-pixel scale on [0,1] normalized coords; "
        "RMSE in pixels = sqrt(MSE) * 96."
    )
    _build_docx(
        "Table T2: Block 1 baselines (best val MSE)",
        caption,
        wide,
        docx_path,
        nd=5,
        footer=footer,
    )
    print(f"T2 -> {csv_path}, {docx_path}")
    return wide


def build_T3(long_df):
    b1 = long_df[long_df["block"] == "01_naive"].copy()
    b2 = long_df[long_df["block"] == "02_dedup"].copy()

    pairs = [
        ("data_large", "large_raw", "large_dedup"),
        ("data_full", "full_raw_clean", "full_dedup"),
    ]

    rows = []
    for disp, before_ds, after_ds in pairs:
        for m in MODELS_ORDER:
            v1 = b1.loc[
                (b1["dataset"] == before_ds) & (b1["model"] == m), "best_val_mse"
            ].values
            v2 = b2.loc[
                (b2["dataset"] == after_ds) & (b2["model"] == m), "best_val_mse"
            ].values
            v1 = v1[0] if len(v1) else None
            v2 = v2[0] if len(v2) else None
            delta = (v2 - v1) if (v1 is not None and v2 is not None) else None
            pct = (
                (delta / v1 * 100)
                if (v1 not in (None, 0) and delta is not None)
                else None
            )
            rows.append(
                {
                    "Dataset": disp,
                    "Model": m,
                    "B1 (leaky)": v1,
                    "B2 (dedup)": v2,
                    "Delta": delta,
                    "Delta %": pct,
                }
            )

    # Sanity check row: data_small (should be identical)
    for m in MODELS_ORDER:
        v1 = b1.loc[
            (b1["dataset"] == "small_raw") & (b1["model"] == m), "best_val_mse"
        ].values
        v1 = v1[0] if len(v1) else None
        rows.append(
            {
                "Dataset": "data_small (sanity)",
                "Model": m,
                "B1 (leaky)": v1,
                "B2 (dedup)": v1,  # reused as-is
                "Delta": 0.0 if v1 is not None else None,
                "Delta %": 0.0 if v1 is not None else None,
            }
        )

    df = pd.DataFrame(rows)

    # format Delta % with 1 decimal + % sign for readability later
    df_display = df.copy()
    df_display["Delta %"] = df_display["Delta %"].apply(
        lambda x: f"{x:+.1f}%" if x is not None and not pd.isna(x) else ""
    )

    csv_path = os.path.join(TABLES_DIR, "T3_leakage_effect.csv")
    docx_path = os.path.join(TABLES_DIR, "T3_leakage_effect.docx")
    df.to_csv(csv_path, index=False)

    caption = (
        "Table T3. Pixel-leakage effect on validation MSE. Before (Block 1, "
        "non-dedup) vs after (Block 2, alpha-rule dedup protecting test>val>train). "
        "Positive Delta means val MSE rose after dedup, i.e. Block 1 was "
        "optimistically biased by cross-split SHA1 duplicates."
    )
    footer = (
        "Note: data_small has zero cross-split duplicates and is therefore "
        "reused unchanged from Block 1 (shown as sanity check). "
        "Cross-split SHA1 duplicates before dedup: data_large 287, "
        "data_full 320, data_small 0."
    )
    _build_docx(
        "Table T3: Pixel leakage effect (Block 1 vs Block 2)",
        caption,
        df_display,
        docx_path,
        nd=5,
        footer=footer,
    )
    print(f"T3 -> {csv_path}, {docx_path}")
    return df


def build_T4(long_df):
    b3 = long_df[long_df["block"] == "03_aug"].copy()

    # PLAN.md calls S1-S5 the augmentation strategies; confirm what Block 3 labels them.
    strategies = sorted(b3["strategy"].unique())

    # Wide: (dataset, model) x strategy
    b3["key"] = b3["dataset"] + "__" + b3["model"]
    wide = b3.pivot_table(
        index=["dataset", "model"],
        columns="strategy",
        values="best_val_mse",
        aggfunc="first",
    )[strategies].reset_index()

    # Order rows
    row_order = [
        (d, m) for d in ["small_raw", "large_dedup", "full_dedup"] for m in MODELS_ORDER
    ]
    wide["__ord__"] = wide.apply(
        lambda r: (
            row_order.index((r["dataset"], r["model"]))
            if (r["dataset"], r["model"]) in row_order
            else 999
        ),
        axis=1,
    )
    wide = wide.sort_values("__ord__").drop(columns="__ord__").reset_index(drop=True)

    # Find best per row and compute relative delta vs S1 baseline
    wide_disp = wide.copy()
    best_col_name = "best_strategy"
    best_vals = []
    best_strats = []
    improve_vs_S1 = []
    for _, r in wide.iterrows():
        strat_vals = {s: r[s] for s in strategies if pd.notna(r[s])}
        if not strat_vals:
            best_vals.append(None)
            best_strats.append("")
            improve_vs_S1.append(None)
            continue
        best_s = min(strat_vals, key=strat_vals.get)
        best_vals.append(strat_vals[best_s])
        best_strats.append(best_s)
        if "S1" in strat_vals and strat_vals["S1"] > 0:
            improve_vs_S1.append(
                (strat_vals[best_s] - strat_vals["S1"]) / strat_vals["S1"] * 100
            )
        else:
            improve_vs_S1.append(None)
    wide_disp["best_strategy"] = best_strats
    wide_disp["best_val_mse"] = best_vals
    wide_disp["pct_vs_S1"] = improve_vs_S1

    csv_path = os.path.join(TABLES_DIR, "T4_block3_factorial.csv")
    docx_path = os.path.join(TABLES_DIR, "T4_block3_factorial.docx")
    wide_disp.to_csv(csv_path, index=False)

    # DOCX display (round best and format pct)
    disp = wide_disp.copy()
    disp["pct_vs_S1"] = disp["pct_vs_S1"].apply(
        lambda x: f"{x:+.1f}%" if x is not None and not pd.isna(x) else ""
    )

    caption = (
        "Table T4. Block 3 full factorial -- best val MSE on clean (dedup) data "
        "across 3 datasets x 3 models x 5 augmentation strategies. "
        "best_strategy is the argmin over S1-S5; pct_vs_S1 reports the "
        "relative change of the best over the S1 (no-aug) baseline."
    )
    footer = (
        "Note: all runs on dedup splits. data_small = 1284/428; "
        "large_dedup = 3948/1360; full_dedup = 3940/1369. "
        "Negative pct_vs_S1 means augmentation helps. "
        "Masked MSE on full_dedup; naive MSE on small/large."
    )
    _build_docx(
        "Table T4: Block 3 augmentation factorial (best val MSE)",
        caption,
        disp,
        docx_path,
        nd=5,
        footer=footer,
    )
    print(f"T4 -> {csv_path}, {docx_path}")
    return wide_disp


def build_T5(long_df, epoch_df):
    # For each (model, strategy) in block 3 on full_dedup, pick the best-val-mse
    # epoch and report cohort mse there.
    b3 = long_df[
        (long_df["block"] == "03_aug") & (long_df["dataset"] == "full_dedup")
    ].copy()
    rows = []
    for _, r in b3.iterrows():
        ed = epoch_df[
            (epoch_df["block"] == "03_aug")
            & (epoch_df["dataset"] == r["dataset"])
            & (epoch_df["model"] == r["model"])
            & (epoch_df["strategy"] == r["strategy"])
            & (epoch_df["seed"] == r["seed"])
        ]
        if ed.empty:
            continue
        best_idx = ed["val_mse"].idxmin()
        be = ed.loc[best_idx]
        rows.append(
            {
                "Model": r["model"],
                "Strategy": r["strategy"],
                "best_val_mse": be["val_mse"],
                "cohort_frontal_mse": be["cohort_frontal_mse"],
                "cohort_asym_mse": be["cohort_asym_mse"],
                "cohort_hard_mse": be["cohort_hard_mse"],
            }
        )
    df = pd.DataFrame(rows)

    # Order: model in MODELS_ORDER, strategy alphabetical (S1..S5)
    df["__ord__"] = df.apply(
        lambda r: (
            MODELS_ORDER.index(r["Model"]) * 10 + int(r["Strategy"][1:])
            if r["Strategy"].startswith("S")
            else 999
        ),
        axis=1,
    )
    df = df.sort_values("__ord__").drop(columns="__ord__").reset_index(drop=True)

    csv_path = os.path.join(TABLES_DIR, "T5_cohort_heterogeneity.csv")
    docx_path = os.path.join(TABLES_DIR, "T5_cohort_heterogeneity.docx")
    df.to_csv(csv_path, index=False)

    caption = (
        "Table T5. Cohort heterogeneity on clean data_full (full_dedup). "
        "Per-cohort validation MSE at the best-val-mse epoch, by model x "
        "augmentation strategy. Cohort cutoffs come from EDA: frontal = "
        "symmetric frontal faces, asym = asymmetric pose, hard = extreme "
        "pose / occlusion (very small n in val)."
    )
    footer = (
        "Note: cohort sizes in val (approximate): frontal n~1200, "
        "asym n~150, hard n<10. cohort_hard_mse values are volatile "
        "because of tiny n; interpret as directional only."
    )
    _build_docx(
        "Table T5: Cohort heterogeneity on full_dedup",
        caption,
        df,
        docx_path,
        nd=5,
        footer=footer,
    )
    print(f"T5 -> {csv_path}, {docx_path}")
    return df


def write_T1_stub():
    path = os.path.join(TABLES_DIR, "T1_legacy_motivation.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "NOTE: T1 legacy motivation table is not generated from the new "
            "60-cell bundle. It is drawn from the pre-restructure 45-run "
            "legacy run at ckpt/baseline_data_aug/ and must be compiled "
            "separately if needed for section 3.0.\n"
        )
    print(f"T1 stub -> {path}")


def _main_build_tables():
    apply_style()
    os.makedirs(TABLES_DIR, exist_ok=True)
    long_df = pd.read_csv(os.path.join(PAPER_DIR, "tier1_full_long.csv"))
    epoch_df = pd.read_csv(os.path.join(PAPER_DIR, "tier1_full_epoch.csv"))

    build_T2(long_df)
    build_T3(long_df)
    build_T4(long_df)
    build_T5(long_df, epoch_df)
    write_T1_stub()
