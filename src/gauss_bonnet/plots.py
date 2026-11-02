"""Plots of the results: the theta_macro distribution and related charts.

The module reads ready ``droplets.csv`` rows (``io.read_droplets_csv``) and recomputes nothing.
Plots:

* ``plot_theta_histogram``: histogram of theta_macro over ``ok`` droplets; the median, mean
  and (if given) true angle are marked with vertical lines;
* ``plot_theta_vs_volume``: theta_macro versus droplet volume (log scale). Colour separates
  droplets with one contact line loop, with several loops and with junctions;
* ``plot_status_counts``: how many droplets and what share of the droplet phase volume (oil
  or water) fall into each status (shows how much of it went to ``border``).

``save_plots`` writes all three files to a folder. Command line:
``gauss-bonnet-plot <folder with droplets.csv> [--true-angle 45]``; ``gauss-bonnet ... --plots``
does the same.

Drawing uses ``matplotlib.figure.Figure`` without ``pyplot``: no GUI backend is needed, and
the functions can be called from threads and on a headless server.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gauss_bonnet import __version__
from gauss_bonnet.io import (
    STATUS_BORDER,
    STATUS_ERROR,
    STATUS_NO_CONTACT,
    STATUS_NOT_SAMPLED,
    STATUS_OK,
    STATUS_OUT_OF_RANGE,
    STATUS_TOO_SMALL,
    read_droplets_csv,
)

log = logging.getLogger("gauss_bonnet")

THETA_HIST_FILE = "theta_hist"
THETA_VS_VOLUME_FILE = "theta_vs_volume"
STATUSES_FILE = "statuses"
PLOT_FORMATS = ("png", "svg", "pdf")
DEFAULT_BIN_WIDTH_DEG = 5.0
DEFAULT_DPI = 150

# Light-theme palette. Categorical colours are the first three slots in a fixed order:
# only these stay pairwise distinguishable under colour blindness.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, teal
MUTED = "#a3a29c"  # "other" statuses

STATUS_LABELS = {
    STATUS_OK: "ok - angle computed",
    STATUS_OUT_OF_RANGE: "out_of_range - θ outside [0°, 180°]",
    STATUS_BORDER: "border - at the sample edge",
    STATUS_TOO_SMALL: "too_small - too few voxels",
    STATUS_NO_CONTACT: "no_contact - no contact with solid",
    STATUS_ERROR: "error - computation failed",
    STATUS_NOT_SAMPLED: "not_sampled - not in the sample",
}


# --------------------------------------------------------------------------- data


def theta_values(rows: Iterable[Mapping[str, Any]], status: str = STATUS_OK) -> np.ndarray:
    """theta_macro (deg) of droplets with the given status; empty values are skipped."""
    return np.array(
        [r["theta_macro_deg"] for r in rows
         if r.get("status") == status and r.get("theta_macro_deg") is not None],
        dtype=np.float64,
    )


def _check_true_angle(true_angle_deg: float | None) -> None:
    if true_angle_deg is not None and not (
        math.isfinite(true_angle_deg) and 0 <= true_angle_deg <= 180
    ):
        raise ValueError(f"true_angle_deg (--true-angle) must be in [0, 180], "
                         f"got {true_angle_deg!r}")


# --------------------------------------------------------------------------- styling


def _new_figure(size=(7.0, 4.2)) -> Figure:
    fig = Figure(figsize=size, facecolor=SURFACE, layout="constrained")
    return fig


def _style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _reference_lines(ax: Axes, values: np.ndarray, true_angle_deg: float | None,
                     *, vertical: bool) -> None:
    """Median, mean and true angle. Labels go to the legend; lines use the text colour."""
    line = ax.axvline if vertical else ax.axhline
    if values.size:
        med, mean = float(np.median(values)), float(values.mean())
        line(med, color=TEXT_PRIMARY, linewidth=2, label=f"median {med:.1f}°")
        line(mean, color=TEXT_PRIMARY, linewidth=1.5, linestyle="--",
             label=f"mean {mean:.1f}°")
    if true_angle_deg is not None:
        line(true_angle_deg, color=TEXT_SECONDARY, linewidth=1.5, linestyle=":",
             label=f"true angle {true_angle_deg:g}°")


def _legend(ax: Axes, **kw) -> None:
    if ax.get_legend_handles_labels()[0]:
        leg = ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, **kw)
        leg.set_zorder(5)


def _empty_note(ax: Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            color=TEXT_SECONDARY, fontsize=11)


# --------------------------------------------------------------------------- plots


def plot_theta_histogram(
    rows: Sequence[Mapping[str, Any]],
    *,
    bin_width_deg: float = DEFAULT_BIN_WIDTH_DEG,
    true_angle_deg: float | None = None,
    title: str | None = None,
) -> Figure:
    """Histogram of theta_macro over ``ok`` droplets on [0, 180] degrees."""
    if not (math.isfinite(bin_width_deg) and 0 < bin_width_deg <= 180):
        raise ValueError(f"bin_width_deg (--bin-width) must be in (0, 180], "
                         f"got {bin_width_deg!r}")
    _check_true_angle(true_angle_deg)
    values = theta_values(rows)
    n_out = theta_values(rows, STATUS_OUT_OF_RANGE).size

    fig = _new_figure()
    ax = fig.add_subplot()
    _style_axes(ax)
    edges = np.arange(0.0, 180.0 + bin_width_deg, bin_width_deg)
    edges = edges[edges <= 180.0 + 1e-9]
    if edges[-1] < 180.0:
        edges = np.append(edges, 180.0)
    if values.size:
        # 2 px gap between bars: an outline in the background colour
        ax.hist(values, bins=edges, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5,
                label="ok droplets")
    else:
        _empty_note(ax, "no droplets with status ok")
    _reference_lines(ax, values, true_angle_deg, vertical=True)

    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel("θ_macro, deg (through water)")
    ax.set_ylabel("number of droplets")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(False, axis="x")
    head = title or "θ_macro distribution"
    sub = f"ok droplets: {values.size}"
    if values.size >= 2:
        sub += f", σ = {values.std(ddof=1):.1f}°"
    if n_out:
        sub += f"; outside [0°, 180°], not shown: {n_out}"
    ax.set_title(f"{head}\n{sub}", fontsize=11, loc="left")
    _legend(ax, loc="upper right")
    return fig


def plot_theta_vs_volume(
    rows: Sequence[Mapping[str, Any]],
    *,
    true_angle_deg: float | None = None,
) -> Figure:
    """theta_macro of ``ok`` droplets versus volume (vox, log scale), split by contact line type."""
    _check_true_angle(true_angle_deg)
    ok = [r for r in rows if r.get("status") == STATUS_OK and r.get("theta_macro_deg") is not None]
    groups = (
        ("one loop (N = 1)",
         lambda r: (r.get("n_loops") or 0) == 1 and not (r.get("n_junctions") or 0)),
        ("several loops (N > 1)",
         lambda r: (r.get("n_loops") or 0) > 1 and not (r.get("n_junctions") or 0)),
        ("line junctions", lambda r: (r.get("n_junctions") or 0) > 0),
    )
    fig = _new_figure((7.0, 4.8))
    ax = fig.add_subplot()
    _style_axes(ax)
    markers =("o", "s", "^")  # shape duplicates colour, so groups read without colour too
    for (label, pick), color, marker in zip(groups, SERIES, markers, strict=True):
        sel = [r for r in ok if pick(r)]
        if not sel:
            continue
        ax.scatter([r["volume_vox"] for r in sel], [r["theta_macro_deg"] for r in sel],
                   s=28, color=color, marker=marker, edgecolors=SURFACE, linewidths=1,
                   alpha=0.9, label=f"{label}: {len(sel)}", zorder=3)
    values = theta_values(ok)
    _reference_lines(ax, values, true_angle_deg, vertical=False)
    if ok:
        ax.set_xscale("log")
    else:
        _empty_note(ax, "no droplets with status ok")
    ax.set_ylim(0, 180)
    ax.set_yticks(np.arange(0, 181, 30))
    ax.set_xlabel("droplet volume, vox (log scale)")
    ax.set_ylabel("θ_macro, deg")
    ax.set_title("θ_macro and droplet size", fontsize=11, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # below the plot so that points near 180° stay visible
        fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False,
                   fontsize=9, labelcolor=TEXT_PRIMARY)
    return fig


def _phase_name(rows: Sequence[Mapping[str, Any]]) -> str:
    phases = {r.get("phase") for r in rows} - {None}
    return {frozenset({"oil"}): "oil", frozenset({"water"}): "water"}.get(
        frozenset(phases), "oil" if not phases else "droplet phase")


def plot_status_counts(rows: Sequence[Mapping[str, Any]]) -> Figure:
    """Droplet count and phase volume share by status (two panels sharing the status axis)."""
    order = [s for s in STATUS_LABELS if any(r.get("status") == s for r in rows)]
    counts = np.array([sum(r.get("status") == s for r in rows) for s in order], dtype=float)
    volume = np.array([sum(r.get("volume_vox") or 0 for r in rows if r.get("status") == s)
                       for s in order], dtype=float)
    total_vol = volume.sum()
    share = 100 * volume / total_vol if total_vol else np.zeros_like(volume)

    fig = _new_figure((8.5, 0.55 * max(len(order), 2) + 1.6))
    ax_n, ax_v = fig.subplots(1, 2, sharey=True)
    for ax in (ax_n, ax_v):
        _style_axes(ax)
        ax.grid(False, axis="y")
    if not order:
        _empty_note(ax_n, "no droplets")
        ax_n.set_yticks([])
        return fig

    y = np.arange(len(order))
    colors = [SERIES[0] if s == STATUS_OK else MUTED for s in order]
    def pct(v: float) -> str:
        return "< 0.1 %" if 0 < v < 0.1 else f"{v:.1f} %"

    for ax, data, fmt, xlabel in (
        (ax_n, counts, "{:.0f}".format, "number of droplets"),
        (ax_v, share, pct, f"{_phase_name(rows)} volume share, %"),
    ):
        ax.barh(y, data, height=0.62, color=colors, edgecolor=SURFACE, linewidth=1)
        top = data.max() if data.size else 1
        for yi, v in zip(y, data, strict=True):
            ax.text(v + 0.01 * top, yi, fmt(v), va="center", ha="left",
                    color=TEXT_PRIMARY, fontsize=9)
        ax.set_xlim(0, top * 1.18 if top else 1)
        ax.set_xlabel(xlabel)
    ax_n.set_yticks(y, [STATUS_LABELS[s] for s in order])
    ax_n.invert_yaxis()
    ax_n.set_title(f"Droplets by status ({len(rows)} total)", fontsize=11, loc="left")
    return fig


# --------------------------------------------------------------------------- output


def save_plots(
    rows: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    true_angle_deg: float | None = None,
    bin_width_deg: float = DEFAULT_BIN_WIDTH_DEG,
    fmt: str = "png",
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Build all plots and write them to ``out_dir``. Return the file paths."""
    if fmt not in PLOT_FORMATS:
        raise ValueError(f"format (--format) must be one of {PLOT_FORMATS}, got {fmt!r}")
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"plot folder points to a file: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        THETA_HIST_FILE: plot_theta_histogram(rows, bin_width_deg=bin_width_deg,
                                              true_angle_deg=true_angle_deg),
        THETA_VS_VOLUME_FILE: plot_theta_vs_volume(rows, true_angle_deg=true_angle_deg),
        STATUSES_FILE: plot_status_counts(rows),
    }
    paths = []
    for name, fig in figures.items():
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, facecolor=SURFACE)
        log.debug("plot: %s", path)
        paths.append(path)
    return paths


def plot_results(results_dir: str | Path, out_dir: str | Path | None = None,
                 **kwargs) -> list[Path]:
    """Read ``droplets.csv`` from ``results_dir`` and write the plots (to the same folder by default)."""
    from gauss_bonnet.cli import DROPLETS_CSV  # imported here to avoid an import cycle

    results_dir = Path(results_dir)
    csv_path = results_dir / DROPLETS_CSV if results_dir.is_dir() else results_dir
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} not found; run gauss-bonnet "
                                f"with --out {results_dir} first")
    rows = read_droplets_csv(csv_path)
    return save_plots(rows, csv_path.parent if out_dir is None else out_dir, **kwargs)


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gauss-bonnet-plot",
        description="Plots of gauss-bonnet results: θ_macro histogram, "
                    "θ_macro versus droplet volume, droplet statuses.",
    )
    p.add_argument("results", help="folder with droplets.csv (or the path to the file)")
    p.add_argument("--out", metavar="DIR", help="where to write the plots (default: the same folder)")
    p.add_argument("--true-angle", type=float, metavar="DEG",
                   help="true angle to mark on the plots for comparison")
    p.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH_DEG, metavar="DEG",
                   help=f"histogram bin width (default {DEFAULT_BIN_WIDTH_DEG:g}°)")
    p.add_argument("--format", choices=PLOT_FORMATS, default="png")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("-q", "--quiet", action="store_true", help="do not print the list of files")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    from gauss_bonnet.cli import EXIT_OK, EXIT_USER, _safe_streams, setup_logging

    _safe_streams()
    args = build_parser().parse_args(argv)
    setup_logging(logging.WARNING if args.quiet else logging.INFO)
    try:
        paths = plot_results(args.results, args.out, true_angle_deg=args.true_angle,
                             bin_width_deg=args.bin_width, fmt=args.format, dpi=args.dpi)
    except (FileNotFoundError, ValueError, OSError) as exc:
        log.error("%s", exc)
        return EXIT_USER
    if not args.quiet:
        for path in paths:
            print(path)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
