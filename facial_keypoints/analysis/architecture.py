"""Analysis / architecture for the facial keypoint study."""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from ..analysis.style import apply_style
from ..paths import FIGURE_DIR

SUBMITTED_FIG = os.path.join(FIGURE_DIR, "F_arch.pdf")


def panel_mlp(ax):
    # 3 visible columns: input (6 nodes, dots below), hidden (5 nodes, dots),
    # output (4 nodes, dots). Use ellipsis to indicate full width.
    col_x = {"in": 0.3, "hid": 1.4, "out": 2.5}
    col_n = {"in": 6, "hid": 5, "out": 4}
    col_c = {"in": "#9fc5e8", "hid": "#c7a4d3", "out": "#a9d18e"}

    def y_positions(n, y_top=1.55, y_bot=-1.55):
        return np.linspace(y_top, y_bot, n)

    nodes = {}
    for k in ("in", "hid", "out"):
        ys = y_positions(col_n[k])
        nodes[k] = [(col_x[k], y) for y in ys]

    # edges: input-hidden, hidden-output
    for xa, ya in nodes["in"]:
        for xb, yb in nodes["hid"]:
            ax.plot([xa, xb], [ya, yb], color="#888", lw=0.35, zorder=1)
    for xa, ya in nodes["hid"]:
        for xb, yb in nodes["out"]:
            ax.plot([xa, xb], [ya, yb], color="#888", lw=0.35, zorder=1)

    # nodes
    for k in ("in", "hid", "out"):
        for x, y in nodes[k]:
            c = Circle(
                (x, y),
                0.11,
                facecolor=col_c[k],
                edgecolor="#222",
                linewidth=0.5,
                zorder=3,
            )
            ax.add_patch(c)

    # ellipsis for "many more nodes"
    for k in ("in", "hid", "out"):
        ax.text(
            col_x[k],
            -1.85,
            r"$\vdots$",
            ha="center",
            va="center",
            fontsize=11,
            color="#333",
        )

    # column labels
    ax.text(
        col_x["in"], 1.95, "input\n$9{,}216$", ha="center", va="bottom", fontsize=7.5
    )
    ax.text(
        col_x["hid"],
        1.95,
        "hidden\n$512$ + ReLU",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    ax.text(col_x["out"], 1.95, "output\n$30$", ha="center", va="bottom", fontsize=7.5)

    ax.set_title("MLP-2", fontsize=10.5, fontweight="bold", pad=6)
    ax.text(
        1.4,
        -2.6,
        "4.72M params $\\cdot$ no spatial prior",
        ha="center",
        va="center",
        fontsize=7.5,
        style="italic",
        color="#555",
    )
    ax.set_xlim(-0.05, 2.85)
    ax.set_ylim(-2.9, 2.55)
    ax.set_axis_off()
    ax.set_aspect("equal")


def cuboid(
    ax,
    x,
    y_center,
    w,
    h,
    depth=0.22,
    fc="#cfe2f3",
    ec="#1f3b66",
    lw=0.6,
    label=None,
    label_below=None,
    label_above=None,
    fontsize=6.8,
):
    """Draw a simple 3-D feature-map cuboid (front face + top/right offset)."""
    y0 = y_center - h / 2
    # front face
    front = Rectangle((x, y0), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(front)
    # top face (parallelogram)
    top = plt.Polygon(
        [
            (x, y0 + h),
            (x + depth, y0 + h + depth),
            (x + w + depth, y0 + h + depth),
            (x + w, y0 + h),
        ],
        facecolor=_darker(fc, 0.12),
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(top)
    # right face
    right = plt.Polygon(
        [
            (x + w, y0),
            (x + w + depth, y0 + depth),
            (x + w + depth, y0 + h + depth),
            (x + w, y0 + h),
        ],
        facecolor=_darker(fc, 0.25),
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(right)
    if label:
        ax.text(
            x + w / 2,
            y_center,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=4,
        )
    if label_below:
        ax.text(
            x + w / 2, y0 - 0.18, label_below, ha="center", va="top", fontsize=fontsize
        )
    if label_above:
        ax.text(
            x + w / 2 + depth / 2,
            y0 + h + depth + 0.08,
            label_above,
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _darker(hex_or_rgb, f=0.2):
    if isinstance(hex_or_rgb, str):
        r, g, b = matplotlib.colors.to_rgb(hex_or_rgb)
    else:
        r, g, b = hex_or_rgb
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def arrow_between(ax, x1, y, x2, color="#333"):
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )


def panel_cnn(ax):
    # Sizes: spatial 96 -> 48 -> 24 -> 12; channels 1 -> 32 -> 64 -> 128
    # Map spatial to height (shrink), channels to width.
    specs = [
        dict(
            x=0.0,
            h=2.2,
            w=0.18,
            ch=1,
            sp=96,
            fc="#eeeeee",
            lab_above="$96^2{\\times}1$",
        ),
        dict(
            x=0.7,
            h=1.5,
            w=0.45,
            ch=32,
            sp=48,
            fc="#9fc5e8",
            lab_above="$48^2{\\times}32$",
        ),
        dict(
            x=1.7,
            h=1.0,
            w=0.70,
            ch=64,
            sp=24,
            fc="#6fa8dc",
            lab_above="$24^2{\\times}64$",
        ),
        dict(
            x=2.95,
            h=0.65,
            w=0.95,
            ch=128,
            sp=12,
            fc="#3d85c6",
            lab_above="$12^2{\\times}128$",
        ),
    ]
    y_c = 0
    for i, s in enumerate(specs):
        cuboid(
            ax,
            s["x"],
            y_c,
            s["w"],
            s["h"],
            depth=0.18,
            fc=s["fc"],
            label_above=s["lab_above"],
        )
        if i > 0:
            x_prev_end = specs[i - 1]["x"] + specs[i - 1]["w"] + 0.18
            arrow_between(ax, x_prev_end, y_c, s["x"] - 0.02)

    # flatten + FC head as thin rectangles
    fx = 4.35
    ax.add_patch(
        Rectangle(
            (fx, -0.55),
            0.18,
            1.1,
            facecolor="#f6f6f6",
            edgecolor="#222",
            linewidth=0.6,
            zorder=3,
        )
    )
    ax.text(
        fx + 0.09, -0.95, "Flatten\n$18{,}432$", ha="center", va="top", fontsize=6.8
    )
    arrow_between(ax, 4.08, y_c, fx - 0.02)

    fc1_x = 5.0
    ax.add_patch(
        Rectangle(
            (fc1_x, -0.45),
            0.18,
            0.9,
            facecolor="#cfe2f3",
            edgecolor="#222",
            linewidth=0.6,
            zorder=3,
        )
    )
    ax.text(fc1_x + 0.09, -0.9, "FC $256$\n+ ReLU", ha="center", va="top", fontsize=6.8)
    arrow_between(ax, fx + 0.18, y_c, fc1_x - 0.02)

    fc2_x = 5.65
    ax.add_patch(
        Rectangle(
            (fc2_x, -0.25),
            0.18,
            0.5,
            facecolor="#a9d18e",
            edgecolor="#222",
            linewidth=0.6,
            zorder=3,
        )
    )
    ax.text(fc2_x + 0.09, -0.55, "FC $30$", ha="center", va="top", fontsize=6.8)
    arrow_between(ax, fc1_x + 0.18, y_c, fc2_x - 0.02)

    ax.set_title("CNN-Shallow", fontsize=10.5, fontweight="bold", pad=18)
    ax.text(
        2.9,
        -2.35,
        "4.75M params $\\cdot$ translation equiv. $\\cdot$ RF $15{\\times}15$",
        ha="center",
        va="center",
        fontsize=7.5,
        style="italic",
        color="#555",
    )
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-2.6, 2.1)
    ax.set_axis_off()
    ax.set_aspect("equal")


def panel_resnet(ax):
    specs = [
        dict(x=0.0, h=2.2, w=0.15, fc="#eeeeee", lab="$96^2{\\times}1$"),
        dict(x=0.55, h=1.6, w=0.40, fc="#f9cb9c", lab="stem $48^2{\\times}64$"),
        dict(
            x=1.3, h=1.3, w=0.55, fc="#6fa8dc", lab="S1 $\\times2$\n$48^2{\\times}64$"
        ),
        dict(
            x=2.3, h=1.0, w=0.7, fc="#3d85c6", lab="S2 $\\times2$\n$24^2{\\times}128$"
        ),
        dict(
            x=3.45,
            h=0.75,
            w=0.85,
            fc="#1f6aa6",
            lab="S3 $\\times2$\n$12^2{\\times}256$",
        ),
        dict(
            x=4.75, h=0.55, w=1.0, fc="#0b3d66", lab="S4 $\\times2$\n$6^2{\\times}512$"
        ),
    ]
    y_c = 0
    centers = []
    for i, s in enumerate(specs):
        cuboid(
            ax,
            s["x"],
            y_c,
            s["w"],
            s["h"],
            depth=0.18,
            fc=s["fc"],
            label_above=s["lab"],
            fontsize=6.3,
        )
        centers.append((s["x"] + s["w"] / 2, y_c, s["x"], s["x"] + s["w"]))
        if i > 0:
            x_prev_end = specs[i - 1]["x"] + specs[i - 1]["w"] + 0.18
            arrow_between(ax, x_prev_end, y_c, s["x"] - 0.02)

    # skip-connection arcs over S1..S4 (indicate residual)
    for i in (2, 3, 4, 5):
        x_start = centers[i][2]
        x_end = centers[i][3]
        y_arc = y_c + specs[i]["h"] / 2 + 0.35
        ax.annotate(
            "",
            xy=(x_end - 0.02, y_c + specs[i]["h"] / 2 - 0.02),
            xytext=(x_start + 0.02, y_c + specs[i]["h"] / 2 - 0.02),
            arrowprops=dict(
                arrowstyle="->",
                color="#c0504d",
                lw=0.55,
                connectionstyle=f"arc3,rad=-0.45",
            ),
        )

    # AvgPool + FC head
    ap_x = 6.0
    ax.add_patch(
        Rectangle(
            (ap_x, -0.30),
            0.18,
            0.6,
            facecolor="#f6f6f6",
            edgecolor="#222",
            linewidth=0.6,
            zorder=3,
        )
    )
    ax.text(ap_x + 0.09, -0.68, "AvgPool\n$512$", ha="center", va="top", fontsize=6.5)
    arrow_between(ax, 5.93, y_c, ap_x - 0.02)

    fc_x = 6.65
    ax.add_patch(
        Rectangle(
            (fc_x, -0.25),
            0.18,
            0.5,
            facecolor="#a9d18e",
            edgecolor="#222",
            linewidth=0.6,
            zorder=3,
        )
    )
    ax.text(fc_x + 0.09, -0.63, "FC $30$", ha="center", va="top", fontsize=6.5)
    arrow_between(ax, ap_x + 0.18, y_c, fc_x - 0.02)

    # residual legend
    ax.plot([5.0, 5.35], [1.9, 1.9], color="#c0504d", lw=0.7)
    ax.text(5.4, 1.9, "residual", ha="left", va="center", fontsize=6.5, color="#c0504d")

    ax.set_title("ResNet-18 (pretrained)", fontsize=10.5, fontweight="bold", pad=18)
    ax.text(
        3.5,
        -2.35,
        "11.2M params $\\cdot$ ImageNet transfer $\\cdot$ residual",
        ha="center",
        va="center",
        fontsize=7.5,
        style="italic",
        color="#555",
    )
    ax.set_xlim(-0.2, 7.3)
    ax.set_ylim(-2.6, 2.2)
    ax.set_axis_off()
    ax.set_aspect("equal")


def _main_build_arch_figure():
    apply_style()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.55, 1.85], wspace=0.02)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_mlp(ax_a)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_cnn(ax_b)
    ax_c = fig.add_subplot(gs[0, 2])
    panel_resnet(ax_c)

    os.makedirs(os.path.dirname(SUBMITTED_FIG), exist_ok=True)
    fig.savefig(SUBMITTED_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"F_arch -> {SUBMITTED_FIG}")
