"""Analysis / style for the facial keypoint study."""

import matplotlib as mpl
import numpy as np

MODEL_COLORS = {
    "mlp2": "#0072B2",  # strong blue
    "cnn_shallow": "#009E73",  # teal-green
    "resnet18_pretrained": "#D55E00",  # vermillion
}

MODEL_LABELS = {
    "mlp2": "MLP-2",
    "cnn_shallow": "CNN-Shallow",
    "resnet18_pretrained": "ResNet-18 (pretr.)",
}

STRATEGY_COLORS = {
    "S1": "#525252",  # dark grey (baseline photometric)
    "S2": "#0072B2",  # blue (geometric)
    "S3": "#009E73",  # green (S1+S2)
    "S4": "#E69F00",  # amber (S3 + scale + blur)
    "S5": "#CC79A7",  # magenta (S3 + erasing)
}

DATASET_COLORS = {
    "small_raw": "#56B4E9",  # sky blue
    "large_raw": "#009E73",
    "full_raw_clean": "#CC79A7",
    "large_dedup": "#009E73",
    "full_dedup": "#CC79A7",
}

DATASET_LABELS = {
    "small_raw": "Small (raw)",
    "large_raw": "Large (raw)",
    "full_raw_clean": "Full (raw, clean)",
    "large_dedup": "Large (dedup)",
    "full_dedup": "Full (dedup)",
}

MODELS_ORDER = ["mlp2", "cnn_shallow", "resnet18_pretrained"]


def apply_style():
    mpl.rcParams.update(
        {
            # typography
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelweight": "regular",
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 11,
            "figure.titleweight": "bold",
            # axes frame
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.6,
            "axes.labelcolor": "#222222",
            # grid: horizontal only, very faint
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#DADADA",
            "grid.linewidth": 0.4,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            # ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            # legend
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "legend.borderaxespad": 0.4,
            # save
            "savefig.bbox": "tight",
            "savefig.dpi": 400,
            "savefig.pad_inches": 0.08,
            "pdf.fonttype": 42,  # embed TrueType (editable text in Illustrator)
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, letter, dx=-0.14, dy=1.04):
    """Add a bold panel letter (a, b, c) in upper-left outside axes."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_value_labels(
    ax, bars, fmt="{:.3f}", dy=0.01, fontsize=7, color="#222", rotation=0
):
    """Annotate bar heights with small italic value labels."""
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=fontsize,
            color=color,
            rotation=rotation,
        )


def despine(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)
