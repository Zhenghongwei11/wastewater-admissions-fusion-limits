from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#7A7A7A",
}


def mm_to_in(mm: float) -> float:
    return float(mm) / 25.4


@dataclass(frozen=True)
class FigureSize:
    width_in: float
    height_in: float


def nature_single_column(*, height_in: float) -> FigureSize:
    # Nature single column width ≈ 89 mm
    return FigureSize(width_in=mm_to_in(89.0), height_in=float(height_in))


def nature_double_column(*, height_in: float) -> FigureSize:
    # Nature double column width ≈ 183 mm
    return FigureSize(width_in=mm_to_in(183.0), height_in=float(height_in))


def apply_journal_style() -> None:
    # Conservative defaults that render well in PDF and are readable at print size.
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "sans-serif",
            # Keep a robust fallback chain across macOS/Linux.
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#3A3A3A",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.color": "#3A3A3A",
            "ytick.color": "#3A3A3A",
            "legend.fontsize": 8,
            "legend.frameon": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=OKABE_ITO["black"],
    )


def save_figure(fig: plt.Figure, *, out_prefix: str, dpi: int = 600) -> tuple[str, str]:
    out_png = f"{out_prefix}.png"
    out_pdf = f"{out_prefix}.pdf"
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    return out_png, out_pdf


def set_tight_limits(ax: plt.Axes, *, pad_frac: float = 0.02) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if math.isfinite(xmin) and math.isfinite(xmax) and xmin != xmax:
        pad = (xmax - xmin) * pad_frac
        ax.set_xlim(xmin - pad, xmax + pad)
    if math.isfinite(ymin) and math.isfinite(ymax) and ymin != ymax:
        pad = (ymax - ymin) * pad_frac
        ax.set_ylim(ymin - pad, ymax + pad)


def set_figure_facecolor(fig: plt.Figure) -> None:
    fig.patch.set_facecolor("white")

