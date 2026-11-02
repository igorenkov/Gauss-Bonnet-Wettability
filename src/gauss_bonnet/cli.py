"""Pipeline of steps 1-6 (Algorithm 1) and the command line: gauss-bonnet <volume.npy> [options].

Droplets are connected components of the ``--droplet-phase`` phase: oil, water or, in ``auto``
mode (the default), whichever of the two has more ``ok`` droplets. ``auto`` is unreliable on
real rock: small clusters of wetting water produce many false "droplets", so choosing water
is accompanied by a warning. The angle is always measured through water. Pipeline per droplet:
``droplets.find_droplets`` -> ``extract_droplet`` -> ``mesh.mesh_droplet`` ->
``topology.mesh_topology`` -> ``curvature.face_curvature`` -> ``contact_angle.contact_angle``.
A mesh is built only for droplets with status ``ok`` after ``find_droplets``. If there are more
of them than ``--max-droplets``, a random sample is computed (``--seed``) and the rest get the
status ``not_sampled``. Other droplets (``border``, ``too_small``, ``no_contact``) are listed
in the table without an angle. If one droplet fails, it gets the status ``error`` (the reason
is logged) and the remaining droplets are still computed.

Output: ``droplets.csv`` (droplets of the chosen phase) and ``summary.json``. The summary holds
statistics over ``ok`` droplets, the chosen phase and totals for each computed phase, the
parameters, ``voxel_size``, the input path and the run time.

Logging goes through the ``gauss_bonnet`` logger. The command line sends it to stderr
(``-v`` verbose, ``-q`` warnings and errors only) and optionally to a file (``--log-file``).
The result is printed to stdout.

Exit codes of ``main``: 0 success; 1 internal error; 2 error in the input data, parameters or
while writing results; 130 interrupted by the user (Ctrl+C).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from gauss_bonnet import __version__
from gauss_bonnet.contact_angle import (
    DEFAULT_EULER,
    DEFAULT_FLAT_EPS,
    DEFAULT_SIGN_CORRECTION,
    EULER_MODES,
    contact_angle,
    droplet_row,
)
from gauss_bonnet.curvature import CURVATURE_METHODS, DEFAULT_N_LAYER, face_curvature
from gauss_bonnet.curvature import DEFAULT_METHOD as DEFAULT_CURVATURE_METHOD
from gauss_bonnet.droplets import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_MIN_VOXELS,
    DEFAULT_PAD,
    extract_droplet,
    find_droplets,
)
from gauss_bonnet.io import (
    STATUS_ERROR,
    STATUS_NOT_SAMPLED,
    STATUS_OK,
    load_volume,
    phase_fractions,
    summarize_droplets,
    validate_volume,
    write_droplets_csv,
    write_summary_json,
)
from gauss_bonnet.mesh import (
    DEFAULT_ALPHA,
    DEFAULT_ITERATIONS,
    SMOOTH_METHODS,
    build_mesh,
    smooth_mesh,
)
from gauss_bonnet.mesh import DEFAULT_METHOD as DEFAULT_SMOOTH_METHOD
from gauss_bonnet.phases import DROPLET_OIL, DROPLET_PHASES, DROPLET_WATER, droplet_volume
from gauss_bonnet.topology import mesh_topology

DROPLETS_CSV = "droplets.csv"
SUMMARY_JSON = "summary.json"
PROGRESS_EVERY = 50  # report progress every N droplets

DEFAULT_SEED = 0
DROPLET_AUTO = "auto"
DROPLET_PHASE_MODES = (DROPLET_AUTO, *DROPLET_PHASES)
DEFAULT_DROPLET_PHASE = DROPLET_AUTO  # may pick the wrong phase on real rock
_PHASE_NAME = {DROPLET_OIL: "oil", DROPLET_WATER: "water"}
_PHASE_LABEL = {DROPLET_OIL: "oil (label 2)", DROPLET_WATER: "water (label 1)"}

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USER = 2
EXIT_INTERRUPTED = 130

log = logging.getLogger("gauss_bonnet")


@dataclass(frozen=True)
class Params:
    """Algorithm parameters. Defaults match the modules.

    Values are validated on creation, so that a parameter error shows up before the volume is
    loaded rather than on the first droplet. The message names both the field and the CLI flag.
    """

    droplet_phase: str = DEFAULT_DROPLET_PHASE
    connectivity: int = DEFAULT_CONNECTIVITY
    min_droplet_voxels: int = DEFAULT_MIN_VOXELS
    triple_line: bool = True
    smooth_method: str = DEFAULT_SMOOTH_METHOD
    smooth_alpha: float = DEFAULT_ALPHA
    smooth_iters: int = DEFAULT_ITERATIONS
    curvature_method: str = DEFAULT_CURVATURE_METHOD
    n_layer: int = DEFAULT_N_LAYER
    sign_correction: bool = DEFAULT_SIGN_CORRECTION
    flat_eps: float = DEFAULT_FLAT_EPS
    euler: str = DEFAULT_EULER
    max_droplets: int | None = None  # None = all droplets; otherwise a random sample
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        problems = []

        def need(ok: bool, field: str, rule: str) -> None:
            if not ok:
                value = getattr(self, field)
                flag = "--" + field.replace("_", "-")
                problems.append(f"{field} ({flag}) {rule}, got {value!r}")

        def finite(x: Any) -> bool:
            return isinstance(x, (int, float)) and math.isfinite(x)

        need(self.droplet_phase in DROPLET_PHASE_MODES, "droplet_phase",
             f"must be one of {DROPLET_PHASE_MODES}")
        need(self.connectivity in (6, 26), "connectivity", "must be 6 or 26")
        need(isinstance(self.min_droplet_voxels, int) and self.min_droplet_voxels >= 1,
             "min_droplet_voxels", "must be an integer >= 1")
        need(self.smooth_method in SMOOTH_METHODS, "smooth_method",
             f"must be one of {SMOOTH_METHODS}")
        need(finite(self.smooth_alpha) and 0 < self.smooth_alpha <= 1, "smooth_alpha",
             "must be in (0, 1]")
        need(isinstance(self.smooth_iters, int) and self.smooth_iters >= 0, "smooth_iters",
             "must be an integer >= 0")
        need(self.curvature_method in CURVATURE_METHODS, "curvature_method",
             f"must be one of {CURVATURE_METHODS}")
        need(isinstance(self.n_layer, int) and self.n_layer >= 1, "n_layer",
             "must be an integer >= 1")
        need(finite(self.flat_eps) and self.flat_eps >= 0, "flat_eps", "must be >= 0")
        need(self.euler in EULER_MODES, "euler", f"must be one of {EULER_MODES}")
        need(self.max_droplets is None
             or (_is_int(self.max_droplets) and self.max_droplets >= 1),
             "max_droplets", "must be an integer >= 1 (or unset)")
        need(_is_int(self.seed) and self.seed >= 0, "seed", "must be an integer >= 0")
        if problems:
            raise ValueError("invalid parameters: " + "; ".join(problems))


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def sample_droplets(droplets, max_droplets: int | None, seed: int, phase: str) -> set[int]:
    """Ids of the ``ok`` droplets to compute: all of them or a random ``max_droplets`` subset.

    Sampling is without replacement, uniform and reproducible: the generator depends only on
    ``seed`` and the phase, so ``auto`` and ``oil``/``water`` modes pick the same droplets.
    """
    ids = [d.id for d in droplets if d.status == STATUS_OK]
    if max_droplets is None or len(ids) <= max_droplets:
        return set(ids)
    rng = np.random.default_rng([seed, DROPLET_PHASES.index(phase)])
    return {int(i) for i in rng.choice(ids, size=max_droplets, replace=False)}


def _check_voxel_size(voxel_size: float) -> None:
    if not (isinstance(voxel_size, (int, float)) and math.isfinite(voxel_size) and voxel_size > 0):
        raise ValueError(f"voxel_size (--voxel-size) must be > 0 um, got {voxel_size!r}")


def _analyze_droplet(volume, labels, droplet, params: Params, voxel_size: float,
                     phase: str) -> dict[str, Any]:
    crop = extract_droplet(volume, labels, droplet, pad=DEFAULT_PAD)
    mesh = smooth_mesh(
        build_mesh(crop, triple_line=params.triple_line),
        alpha=params.smooth_alpha,
        iterations=params.smooth_iters,
        method=params.smooth_method,
    )
    topo = mesh_topology(mesh)
    curv = face_curvature(mesh, params.n_layer, method=params.curvature_method)
    angle = contact_angle(
        mesh,
        topo,
        curv,
        sign_correction=params.sign_correction,
        flat_eps=params.flat_eps,
        euler=params.euler,
    )
    row = droplet_row(droplet, mesh, topo, angle, voxel_size=voxel_size, phase=phase)
    theta = row["theta_macro_deg"]
    log.debug(
        "droplet %s %d: %d vox, N=%d, χ(∂D)=%d, θ_macro=%s, status %s",
        _PHASE_LABEL[phase], droplet.id, droplet.volume_vox, topo.n_loops, topo.euler_surface,
        "-" if theta is None else f"{theta:.1f}°", angle.status,
    )
    return row


def analyze_volume(
    volume: np.ndarray,
    params: Params | None = None,
    *,
    voxel_size: float = 1.0,
    progress=None,
) -> list[dict[str, Any]]:
    """Compute theta_macro for all droplets of the chosen phase. Returns ``droplets.csv`` rows.

    Same as ``analyze_phases`` but without the phase selection details.
    """
    return analyze_phases(volume, params, voxel_size=voxel_size, progress=progress)[0]


def analyze_phases(
    volume: np.ndarray,
    params: Params | None = None,
    *,
    voxel_size: float = 1.0,
    progress=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute droplets of the phase ``params.droplet_phase``; in ``auto`` mode, of both phases.

    In ``auto`` mode the phase with more ``ok`` droplets wins (oil on a tie). This assumes it
    is the non-wetting phase, but on real rock wetting water breaks into thousands of small
    clusters and wins by count. If ``max_droplets`` is set and there are more valid droplets,
    a random sample is computed, and the ``ok`` count used for the choice is estimated from it:
    n_ok * n_candidates / n_sampled.

    Returns
    -------
    rows
        ``droplets.csv`` rows for the chosen phase.
    selection
        ``{"droplet_phase": phase, "mode": mode, "phases": {phase: summary of its droplets}}``;
        each summary is as in ``io.summarize_droplets``, plus ``n_candidates`` (``ok`` droplets
        before computation), ``n_sampled`` (computed) and ``n_droplets_ok_estimated`` (estimated
        number of ``ok`` droplets among all candidates).

    ``progress(i, n)`` is an optional callback after each processed droplet (in ``auto`` mode
    n counts droplets of both phases). A droplet whose computation fails gets the status
    ``error``; the exception is logged as a warning (with a traceback at DEBUG level) and the
    run continues.
    """
    params = Params() if params is None else params
    _check_voxel_size(voxel_size)
    volume = validate_volume(volume)
    mode = params.droplet_phase
    phases = DROPLET_PHASES if mode == DROPLET_AUTO else (mode,)

    found = {}
    for phase in phases:
        vol = droplet_volume(volume, phase)
        labels, droplets = find_droplets(
            vol, connectivity=params.connectivity, min_voxels=params.min_droplet_voxels
        )
        chosen_ids = sample_droplets(droplets, params.max_droplets, params.seed, phase)
        found[phase] = (vol, labels, droplets, chosen_ids)
        n_todo = sum(d.status == STATUS_OK for d in droplets)
        if not droplets:
            log.warning("no %s in the volume: no droplets found", _PHASE_LABEL[phase])
        else:
            log.info("%s droplets: %d found, %d to compute",
                     _PHASE_LABEL[phase], len(droplets), n_todo)
        if len(chosen_ids) < n_todo:
            log.info("%s droplets: --max-droplets computes a random sample of %d out of %d "
                     "(--seed %d), the rest get status not_sampled",
                     _PHASE_LABEL[phase], len(chosen_ids), n_todo, params.seed)

    total = sum(len(f[2]) for f in found.values())
    done = 0
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for phase, (vol, labels, droplets, chosen_ids) in found.items():
        rows = []
        for d in droplets:
            if d.status == STATUS_OK and d.id not in chosen_ids:
                rows.append({**droplet_row(d, voxel_size=voxel_size, phase=phase),
                             "status": STATUS_NOT_SAMPLED})
            else:
                rows.append(_droplet_or_error(vol, labels, d, params, voxel_size, phase))
            done += 1
            if progress is not None:
                progress(done, total)
        by_phase[phase] = rows

    summaries = {}
    for phase, rows in by_phase.items():
        sm = summarize_droplets(rows)
        n_cand = len(found[phase][3]) + sm["status_counts"][STATUS_NOT_SAMPLED]
        n_sampled = len(found[phase][3])
        sm.update(n_candidates=n_cand, n_sampled=n_sampled,
                  n_droplets_ok_estimated=(sm["n_droplets_ok"] * n_cand / n_sampled
                                           if n_sampled else 0.0))
        summaries[phase] = sm
    # ties go to the first phase (oil); without sampling the estimate is the exact ok count
    chosen = max(phases, key=lambda ph: summaries[ph]["n_droplets_ok_estimated"])
    if mode == DROPLET_AUTO:
        log.info("droplet phase selection: %s; chosen %s",
                 "; ".join(f"{_PHASE_NAME[ph]}: ok droplets {_fmt_ok(sm)}"
                           f"{_fmt_median(sm['theta_median_deg'])}"
                           for ph, sm in summaries.items()),
                 _PHASE_NAME[chosen])
        if chosen == DROPLET_WATER:
            log.warning("auto mode chose water droplets. On real rock these are often small "
                        "clusters of wetting water, and the angle comes out falsely hydrophobic; "
                        "compare with --droplet-phase oil")

    rows = by_phase[chosen]
    n_err = sum(r["status"] == STATUS_ERROR for r in rows)
    if n_err:
        log.warning("droplets with status error: %d (run with -v for details)", n_err)
    return rows, {"droplet_phase": chosen, "mode": mode, "phases": summaries}


def _droplet_or_error(volume, labels, d, params: Params, voxel_size: float,
                      phase: str) -> dict[str, Any]:
    if d.status != STATUS_OK:
        return droplet_row(d, voxel_size=voxel_size, phase=phase)
    try:
        return _analyze_droplet(volume, labels, d, params, voxel_size, phase)
    except Exception as exc:  # one bad droplet must not stop the whole run
        log.warning("droplet %s %d (%d vox, centre z/y/x = %s): computation failed - %s: %s",
                    _PHASE_LABEL[phase], d.id, d.volume_vox, _fmt_point(d.centroid),
                    type(exc).__name__, exc)
        log.debug("traceback for droplet %d", d.id, exc_info=True)
        return {**droplet_row(d, voxel_size=voxel_size, phase=phase), "status": STATUS_ERROR}


def _fmt_ok(sm: dict[str, Any]) -> str:
    if sm["n_sampled"] == sm["n_candidates"]:
        return str(sm["n_droplets_ok"])
    return (f"{sm['n_droplets_ok']} of {sm['n_sampled']} sampled "
            f"(~{sm['n_droplets_ok_estimated']:.0f} of {sm['n_candidates']})")


def _fmt_median(median: float | None) -> str:
    return "" if median is None else f", median {median:.1f}°"


def _fmt_point(p) -> str:
    return "/".join(f"{float(c):.0f}" for c in p)


def _check_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"output folder (--out) points to a file: {out_dir}")


def run(
    volume_path: str | Path,
    out_dir: str | Path,
    params: Params | None = None,
    *,
    voxel_size: float = 1.0,
    progress=None,
) -> dict[str, Any]:
    """Load the volume, compute, write ``droplets.csv`` and ``summary.json``. Return the summary."""
    _check_voxel_size(voxel_size)
    params = Params() if params is None else params
    out_dir = Path(out_dir)
    _check_out_dir(out_dir)  # before the long computation, not after
    t0 = time.perf_counter()

    log.info("loading %s", volume_path)
    volume = load_volume(volume_path)
    fractions = phase_fractions(volume)
    log.info("volume %s (z, y, x); phase fractions: solid %.1f %%, water %.1f %%, oil %.1f %%",
             "x".join(map(str, volume.shape)), 100 * fractions[0], 100 * fractions[1],
             100 * fractions[2])
    log.debug("parameters: %s, voxel_size = %g um", asdict(params), voxel_size)

    rows, selection = analyze_phases(volume, params, voxel_size=voxel_size, progress=progress)
    csv_path = write_droplets_csv(rows, out_dir / DROPLETS_CSV)

    summary = summarize_droplets(rows)
    summary.update(
        droplet_phase=selection["droplet_phase"],
        phase_selection={"mode": selection["mode"], "phases": {
            ph: {k: sm[k] for k in ("n_droplets_total", "n_candidates", "n_sampled",
                                    "n_droplets_ok", "n_droplets_ok_estimated",
                                    "theta_median_deg")}
            for ph, sm in selection["phases"].items()}},
        input=str(volume_path),
        shape=list(volume.shape),
        phase_fractions={name: fractions[label] for name, label in
                         (("solid", 0), ("water", 1), ("oil", 2))},
        voxel_size_um=voxel_size,
        params=asdict(params),
        version=__version__,
        elapsed_s=round(time.perf_counter() - t0, 3),
    )
    json_path = write_summary_json(summary, out_dir / SUMMARY_JSON)
    if rows and summary["n_droplets_ok"] == 0:
        log.warning("no droplet got status ok: the θ_macro median is undefined "
                    "(statuses: %s)", {k: v for k, v in summary["status_counts"].items() if v})
    log.debug("written: %s, %s (%.1f s)", csv_path, json_path, summary["elapsed_s"])
    return summary


# --------------------------------------------------------------------------- command line


def _bool(text: str) -> bool:
    low = text.lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"expected a number > 0, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    d = Params()
    p = argparse.ArgumentParser(
        prog="gauss-bonnet",
        description="Macroscopic contact angle (through water) from oil or water droplets "
                    "in a segmented 3D rock image (Eq. 20, Sun et al. 2020).",
        epilog="Exit codes: 0 success, 1 internal error, 2 error in input data "
               "or parameters, 130 interrupted.",
    )
    p.add_argument("volume", help="path to a 3D .npy array (0 = solid, 1 = water, 2 = oil)")
    p.add_argument("--voxel-size", type=_positive_float, default=1.0, metavar="UM",
                   help="voxel size, um; affects only areas in um^2 (default 1)")
    p.add_argument("--out", default="results", metavar="DIR",
                   help="folder for droplets.csv and summary.json (default results)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    v = p.add_argument_group("output and logging")
    level = v.add_mutually_exclusive_group()
    level.add_argument("-v", "--verbose", action="store_true",
                       help="verbose log: every droplet, error tracebacks")
    level.add_argument("-q", "--quiet", action="store_true",
                       help="warnings and errors only, no result on stdout")
    v.add_argument("--log-file", metavar="FILE",
                   help="append a verbose log (DEBUG level) to this file")
    v.add_argument("--plots", action="store_true",
                   help="draw plots in the --out folder (see gauss-bonnet-plot --help)")
    v.add_argument("--true-angle", type=float, metavar="DEG",
                   help="true angle to mark on the plots (only with --plots)")

    g = p.add_argument_group("algorithm parameters")
    g.add_argument("--droplet-phase", choices=DROPLET_PHASE_MODES, default=d.droplet_phase,
                   help="which phase forms the droplets (default auto: the one with more "
                        "ok droplets; may pick the wrong phase on real rock). "
                        "The angle is always through water")
    g.add_argument("--max-droplets", type=int, default=d.max_droplets, metavar="N",
                   help="compute at most N droplets per phase: a random sample of valid ones, "
                        "the rest get status not_sampled (default: all)")
    g.add_argument("--seed", type=int, default=d.seed, metavar="S",
                   help="random seed for --max-droplets (default 0)")
    g.add_argument("--connectivity", type=int, choices=(6, 26), default=d.connectivity,
                   help="voxel connectivity used to find droplets")
    g.add_argument("--min-droplet-voxels", type=int, default=d.min_droplet_voxels, metavar="N",
                   help="droplets smaller than N voxels get status too_small")
    g.add_argument("--triple-line", type=_bool, default=d.triple_line, metavar="BOOL",
                   help="triple line position correction")
    g.add_argument("--smooth-method", choices=SMOOTH_METHODS, default=d.smooth_method)
    g.add_argument("--smooth-alpha", type=float, default=d.smooth_alpha, metavar="A",
                   help="smoothing step, (0, 1]")
    g.add_argument("--smooth-iters", type=int, default=d.smooth_iters, metavar="N")
    g.add_argument("--curvature-method", choices=CURVATURE_METHODS, default=d.curvature_method)
    g.add_argument("--n-layer", type=int, default=d.n_layer, metavar="N",
                   help="neighbour rings for the quadric fit")
    g.add_argument("--sign-correction", type=_bool, default=d.sign_correction, metavar="BOOL",
                   help="sign correction of κ_G·A from Algorithm 1")
    g.add_argument("--flat-eps", type=float, default=d.flat_eps, metavar="EPS")
    g.add_argument("--euler", choices=EULER_MODES, default=d.euler,
                   help="χ(D) from the mesh or χ(D) = 1")
    return p


class _ConsoleFormatter(logging.Formatter):
    """INFO is plain text, other levels get a prefix (``warning: ...``)."""

    LABELS: ClassVar[dict[int, str]] = {logging.DEBUG: "debug", logging.WARNING: "warning",
              logging.ERROR: "error", logging.CRITICAL: "error"}

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        label = self.LABELS.get(record.levelno)
        return text if label is None else f"{label}: {text}"


_HANDLER_TAG = "_gauss_bonnet_cli"


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Set up the ``gauss_bonnet`` logger: stderr at ``level`` and, if given, a file (DEBUG).

    A repeated call replaces the handlers installed by the previous one.
    """
    for h in list(log.handlers):
        if getattr(h, _HANDLER_TAG, False):
            log.removeHandler(h)
            h.close()
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(_ConsoleFormatter("%(message)s"))
    handlers: list[logging.Handler] = [console]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        handlers.append(fh)
    for h in handlers:
        setattr(h, _HANDLER_TAG, True)
        log.addHandler(h)
    log.setLevel(min(h.level for h in handlers))


def _safe_streams() -> None:
    """Do not crash on characters like θ when the console is cp1251 or output is redirected."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def _print_result(summary: dict[str, Any], out: Path) -> None:
    counts = ", ".join(f"{k} {v}" for k, v in summary["status_counts"].items() if v)
    print(f"droplets: {_PHASE_NAME[summary['droplet_phase']]}; "
          f"total {summary['n_droplets_total']} ({counts or 'none'})")
    if summary["theta_median_deg"] is not None:
        std = summary["theta_std_deg"]
        std_text = "" if std is None else f" ± {std:.1f}°"
        print(f"θ_macro ({summary['n_droplets_ok']} ok droplets): "
              f"median {summary['theta_median_deg']:.1f}°, "
              f"mean {summary['theta_mean_deg']:.1f}°{std_text}")
    else:
        print("θ_macro: no droplets with status ok")
    print(f"results: {out / DROPLETS_CSV}, {out / SUMMARY_JSON} ({summary['elapsed_s']} s)")


def main(argv: list[str] | None = None) -> int:
    _safe_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    try:
        setup_logging(level, args.log_file)
    except OSError as exc:
        parser.error(f"cannot open log file {args.log_file}: {exc}")

    try:
        params = Params(
            droplet_phase=args.droplet_phase,
            connectivity=args.connectivity,
            min_droplet_voxels=args.min_droplet_voxels,
            triple_line=args.triple_line,
            smooth_method=args.smooth_method,
            smooth_alpha=args.smooth_alpha,
            smooth_iters=args.smooth_iters,
            curvature_method=args.curvature_method,
            n_layer=args.n_layer,
            sign_correction=args.sign_correction,
            flat_eps=args.flat_eps,
            euler=args.euler,
            max_droplets=args.max_droplets,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))  # exit code 2 and usage hint
    if args.true_angle is not None:
        if not args.plots:
            parser.error("--true-angle is only used together with --plots")
        if not (math.isfinite(args.true_angle) and 0 <= args.true_angle <= 180):
            parser.error(f"--true-angle must be in [0, 180] deg, got {args.true_angle}")

    def progress(i: int, n: int) -> None:
        if i == n or i % PROGRESS_EVERY == 0:
            log.info("droplets processed: %d/%d", i, n)

    log.debug("gauss-bonnet %s, Python %s", __version__, sys.version.split()[0])
    try:
        summary = run(args.volume, args.out, params, voxel_size=args.voxel_size,
                      progress=progress)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return EXIT_USER
    except ValueError as exc:
        log.error("%s", exc)
        return EXIT_USER
    except MemoryError:
        log.error("out of memory while processing %s; close other programs "
                  "or split the volume into parts", args.volume)
        return EXIT_INTERNAL
    except OSError as exc:
        log.error("I/O error: %s", exc)
        return EXIT_USER
    except KeyboardInterrupt:
        log.error("interrupted by the user; results not written")
        return EXIT_INTERRUPTED
    except Exception as exc:  # last resort: a readable message instead of a traceback
        log.error("internal error: %s: %s", type(exc).__name__, exc)
        log.debug("traceback", exc_info=True)
        if not args.verbose:
            log.error("rerun with -v or --log-file to see the traceback")
        return EXIT_INTERNAL

    if not args.quiet:
        _print_result(summary, Path(args.out))
    if args.plots:
        from gauss_bonnet.plots import plot_results  # matplotlib is loaded only on demand

        try:
            paths = plot_results(args.out, true_angle_deg=args.true_angle)
        except (ValueError, OSError) as exc:
            log.error("plots not drawn (results were written): %s", exc)
            return EXIT_USER
        if not args.quiet:
            print("plots: " + ", ".join(str(p) for p in paths))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
