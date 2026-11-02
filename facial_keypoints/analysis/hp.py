"""Analysis / hp for the facial keypoint study."""

import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from ..analysis.style import MODEL_COLORS, MODEL_LABELS, MODELS_ORDER, apply_style
from ..paths import FIGURE_DIR
from ..paths import MASTER_LOG as MASTER

OUT_FIG = os.path.join(FIGURE_DIR, "F20_hp_search.pdf")


TRIAL_RE = re.compile(
    r"\[(?P<model>\w+)\]\s+search_\w+_trial(?P<trial>\d+)\s+"
    r"val=(?P<val>[0-9.]+)\s+test=(?P<test>[0-9.]+)\s+"
    r"test_pix=(?P<test_pix>[0-9.]+)"
)


def parse_master():
    rows = []
    with open(MASTER) as f:
        for line in f:
            m = TRIAL_RE.search(line)
            if not m:
                continue
            rows.append(
                {
                    "model": m["model"],
                    "trial": int(m["trial"]),
                    "val_mse": float(m["val"]),
                    "test_pix": float(m["test_pix"]),
                }
            )
    return rows


def lighten(hex_color, f=0.55):
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def darken(hex_color, f=0.35):
    r, g, b = to_rgb(hex_color)
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def _main_build_hp_figure():
    apply_style()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    rows = parse_master()
    by_model = {m: [r for r in rows if r["model"] == m] for m in MODELS_ORDER}
    for m in MODELS_ORDER:
        by_model[m].sort(key=lambda r: r["trial"])
        assert len(by_model[m]) == 8, f"expected 8 trials for {m}"

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.6),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.35},
    )

    for m in MODELS_ORDER:
        trials = by_model[m]
        xs = [r["trial"] for r in trials]
        vals = [math.sqrt(r["val_mse"]) * 48.0 for r in trials]
        ax_a.scatter(
            xs,
            vals,
            color=MODEL_COLORS[m],
            label=MODEL_LABELS[m],
            s=28,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        best_idx = int(np.argmin(vals))
        ax_a.scatter(
            [xs[best_idx]],
            [vals[best_idx]],
            s=70,
            facecolor="white",
            edgecolor=MODEL_COLORS[m],
            linewidth=1.3,
            zorder=4,
        )

    ax_a.set_xlabel("Trial index")
    ax_a.set_ylabel("Val pixel RMSE")
    ax_a.set_xticks(range(1, 9))
    ax_a.set_title("(a) Per-trial val RMSE", fontsize=9)
    ax_a.legend(loc="upper right", fontsize=7, frameon=False)

    width = 0.36
    gap = 0.14
    xs_b = []
    for i, m in enumerate(MODELS_ORDER):
        trials = by_model[m]
        vals = [math.sqrt(r["val_mse"]) * 48.0 for r in trials]
        best = int(np.argmin(vals))
        val_pix = vals[best]
        test_pix = trials[best]["test_pix"]
        base = MODEL_COLORS[m]
        x_center = i * (2 * width + gap + 0.12)
        x_val = x_center
        x_test = x_center + width
        xs_b.append(x_center + width / 2)

        ax_b.bar(
            x_val,
            val_pix,
            width,
            color=lighten(base, 0.55),
            edgecolor=base,
            linewidth=0.8,
            label=f"{MODEL_LABELS[m]} --- Val",
        )
        ax_b.bar(
            x_test,
            test_pix,
            width,
            color=darken(base, 0.0),
            edgecolor="#222",
            linewidth=0.6,
            label=f"{MODEL_LABELS[m]} --- Test",
        )

        ax_b.text(
            x_val,
            val_pix + 0.05,
            f"{val_pix:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
        ax_b.text(
            x_test,
            test_pix + 0.05,
            f"{test_pix:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
        ax_b.text(x_val, -0.18, "Val", ha="center", va="top", fontsize=7)
        ax_b.text(x_test, -0.18, "Test", ha="center", va="top", fontsize=7)

    ax_b.set_xticks(xs_b)
    ax_b.set_xticklabels([MODEL_LABELS[m] for m in MODELS_ORDER], fontsize=8)
    ax_b.tick_params(axis="x", pad=14)
    ax_b.set_ylabel("Pixel RMSE")
    ax_b.set_ylim(0, max(4.5, max(ax_b.get_ylim())))
    ax_b.set_title("(b) Best-trial val and test pixel RMSE", fontsize=9)
    ax_b.legend(
        loc="upper right",
        fontsize=6,
        frameon=False,
        ncol=1,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.25,
    )

    for ax in (ax_a, ax_b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"F20 -> {OUT_FIG}")
