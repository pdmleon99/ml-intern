"""One shared visual identity for every chart the pipeline generates — before this, each
chart module picked its own ad hoc colors ("steelblue", "coral", "#3498db"...) with default
matplotlib chrome (heavy borders, cramped ticks), so the report looked like a assortment of
unrelated tutorial plots instead of one product. Every viz module imports `apply_style()` at
import time and uses `PALETTE`/`style_ax()` instead of picking its own colors.
"""

import matplotlib
import matplotlib.pyplot as plt

# A cohesive palette (indigo/violet primary, slate neutrals) instead of matplotlib's
# default tab10 cycle or scattered CSS-color-name choices.
PALETTE = {
    "primary": "#4f46e5",       # indigo-600
    "primary_light": "#a5b4fc",  # indigo-300
    "secondary": "#7c3aed",     # violet-600
    "accent": "#0ea5e9",        # sky-500
    "success": "#059669",       # emerald-600
    "warning": "#d97706",       # amber-600
    "danger": "#e11d48",        # rose-600
    "neutral": "#64748b",       # slate-500
    "neutral_light": "#cbd5e1",  # slate-300
    "grid": "#e2e8f0",          # slate-200
    "text": "#1e293b",          # slate-800
    "text_muted": "#64748b",    # slate-500
}

# A perceptually distinct, on-brand sequence for multi-series charts (model comparisons,
# categorical breakdowns) — replaces matplotlib's default tab10 cycle.
SEQUENCE = ["#4f46e5", "#0ea5e9", "#059669", "#d97706", "#e11d48", "#7c3aed", "#64748b"]

DIVERGING_CMAP = "coolwarm"  # kept for correlation heatmaps — diverging semantics matter more than brand color there


def apply_style() -> None:
    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["neutral_light"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "text.color": PALETTE["text"],
        "xtick.color": PALETTE["text_muted"],
        "ytick.color": PALETTE["text_muted"],
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"],
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 100,
        "savefig.dpi": 130,
        "savefig.facecolor": "white",
    })


def style_ax(ax) -> None:
    """Strip chart junk: no top/right spine, light remaining spines, subtle tick marks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["neutral_light"])
    ax.spines["bottom"].set_color(PALETTE["neutral_light"])
    ax.tick_params(length=0)
