from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch

from figures.style import OKABE_ITO
from figures.style import apply_journal_style
from figures.style import nature_double_column
from figures.style import save_figure
from figures.style import set_figure_facecolor

REPO_ROOT = Path(__file__).resolve().parents[1]


def _box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    face: str = "#FFFFFF",
    edge: str = "#C9D1DA",
    dashed: bool = False,
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=edge,
        linestyle="--" if dashed else "-",
        facecolor=face,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="600",
        color="#111827",
        zorder=3,
        linespacing=1.15,
    )
    return x, y, w, h


def _arrow(
    ax: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    dashed: bool = False,
    color: str = "#6B7280",
) -> None:
    a = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        linestyle="--" if dashed else "-",
        color=color,
        zorder=1,
    )
    ax.add_patch(a)


def _midright(node: tuple[float, float, float, float]) -> tuple[float, float]:
    return (node[0] + node[2], node[1] + node[3] / 2)


def _midleft(node: tuple[float, float, float, float]) -> tuple[float, float]:
    return (node[0], node[1] + node[3] / 2)


def _midtop(node: tuple[float, float, float, float]) -> tuple[float, float]:
    return (node[0] + node[2] / 2, node[1] + node[3])


def _midbottom(node: tuple[float, float, float, float]) -> tuple[float, float]:
    return (node[0] + node[2] / 2, node[1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Render Figure 1 study-overview schematic (minimal journal style).")
    ap.add_argument(
        "--out-prefix",
        default="plots/fig1_study_overview",
        help="Output prefix for PNG/PDF (default: plots/fig1_study_overview).",
    )
    args = ap.parse_args()

    apply_journal_style()
    size = nature_double_column(height_in=2.20)
    fig, ax = plt.subplots(figsize=(size.width_in, size.height_in))
    set_figure_facecolor(fig)
    ax.set_axis_off()

    ww = _box(ax, x=0.06, y=0.64, w=0.26, h=0.18, label="Wastewater\n(NWSS)")
    hosp = _box(ax, x=0.06, y=0.40, w=0.26, h=0.18, label="Admissions\n(NHSN)")
    rev = _box(ax, x=0.06, y=0.16, w=0.26, h=0.18, label="Revision audit\n(NYC example)", dashed=True)

    panel = _box(ax, x=0.38, y=0.52, w=0.24, h=0.18, label="Weekly panel\n(aligned)")
    eval_state = _box(
        ax,
        x=0.38,
        y=0.28,
        w=0.24,
        h=0.18,
        label="State-scale evaluation\n(reporting constraints)",
        face="#FBFCFF",
        edge="#9CB7E6",
    )

    ed = _box(ax, x=0.70, y=0.52, w=0.24, h=0.18, label="Positive control\n(ED subset)")
    stable = _box(ax, x=0.70, y=0.28, w=0.24, h=0.18, label="Real-time readiness\n(stability window)")
    concl = _box(
        ax,
        x=0.70,
        y=0.04,
        w=0.24,
        h=0.18,
        label="Scale dependence\n(state vs ED)",
        edge=OKABE_ITO["gray"],
    )

    _arrow(ax, start=_midright(ww), end=_midleft(panel))
    _arrow(ax, start=_midright(hosp), end=_midleft(panel))
    _arrow(ax, start=_midbottom(panel), end=_midtop(eval_state))
    _arrow(ax, start=_midright(eval_state), end=_midleft(ed))
    _arrow(ax, start=_midright(eval_state), end=_midleft(stable))
    _arrow(ax, start=_midright(rev), end=_midleft(stable), dashed=True)
    _arrow(ax, start=_midbottom(ed), end=_midtop(concl))
    _arrow(ax, start=_midbottom(stable), end=_midtop(concl))

    out_prefix = (REPO_ROOT / str(args.out_prefix)).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png, out_pdf = save_figure(fig, out_prefix=str(out_prefix), dpi=600)
    plt.close(fig)

    print(f"OK fig1 overview: wrote {Path(out_png).relative_to(REPO_ROOT)} and {Path(out_pdf).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

