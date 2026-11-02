"""Input and output: loading a 3D .npy volume, writing droplets.csv and summary.json.

Input format: a 3D array, axes (z, y, x), labels 0 = solid, 1 = water, 2 = oil.
Integer and floating dtypes are accepted (e.g. float64), but values must be exactly 0, 1 or 2.
Inside the package the volume is always uint8.

Output format: a per-droplet table ``droplets.csv`` with the fixed column set
``DROPLET_COLUMNS`` and a summary ``summary.json``.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gauss_bonnet.phases import ALL_PHASES, DROPLET_PHASES, OIL

MIN_SIDE = 3  # minimum volume size along each axis, vox

# droplets.csv columns in output order and their types. Units: lengths/areas in voxels,
# *_um2 in um^2, deficit curvature in rad, *_deg in degrees.
# phase is the droplet phase (oil or water); *_deg angles are always measured through water,
# while areas and the deficit curvature refer to the droplet surface (lv = droplet-other fluid).
DROPLET_COLUMNS: dict[str, type] = {
    "id": int,
    "phase": str,
    "volume_vox": int,
    "centroid_z": float,
    "centroid_y": float,
    "centroid_x": float,
    "touches_border": bool,
    "n_loops": int,
    "n_junctions": int,
    "euler_surface": int,
    "euler_droplet": float,
    "area_lv": float,
    "area_ls": float,
    "area_lv_um2": float,
    "area_ls_um2": float,
    "sum_kappa_g_area_lv": float,
    "sum_kappa_g_area_ls": float,
    "deficit_curvature": float,
    "deficit_curvature_raw": float,
    "theta_macro_deg": float,
    "theta_macro_raw_deg": float,
    "status": str,
}

STATUS_OK = "ok"
STATUS_NO_CONTACT = "no_contact"  # droplet does not touch the solid (N = 0)
STATUS_BORDER = "border"  # droplet touches the image boundary
STATUS_TOO_SMALL = "too_small"  # smaller than min_voxels
STATUS_OUT_OF_RANGE = "out_of_range"  # theta_macro outside [0, pi]
STATUS_ERROR = "error"  # droplet computation failed; the reason is logged (cli.analyze_volume)
STATUS_NOT_SAMPLED = "not_sampled"  # valid droplet left out of the random sample (--max-droplets)
STATUSES = frozenset(
    {
        STATUS_OK,
        STATUS_NOT_SAMPLED,
        STATUS_NO_CONTACT,
        STATUS_BORDER,
        STATUS_TOO_SMALL,
        STATUS_OUT_OF_RANGE,
        STATUS_ERROR,
    }
)


# --------------------------------------------------------------------------- input


def validate_volume(volume: np.ndarray) -> np.ndarray:
    """Validate a labelled volume and return it as uint8 (C order).

    Raises
    ------
    ValueError
        Not 3D; a side below ``MIN_SIDE``; non-numeric dtype; values outside {0, 1, 2}
        (including fractional values and NaN).
    """
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D array, got ndim={arr.ndim}, shape={arr.shape}")
    if min(arr.shape) < MIN_SIDE:
        raise ValueError(f"each side must be >= {MIN_SIDE} vox, shape={arr.shape}")
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"expected an integer or float dtype, got {arr.dtype}")

    valid = np.zeros(arr.shape, dtype=bool)
    for label in ALL_PHASES:
        valid |= arr == label
    if not valid.all():
        bad = np.unique(arr[~valid])
        shown = ", ".join(repr(v) for v in bad[:10].tolist())
        more = f" and {bad.size - 10} more" if bad.size > 10 else ""
        raise ValueError(
            f"invalid labels ({int((~valid).sum())} vox): {shown}{more}; "
            f"only {ALL_PHASES} are allowed (0 = solid, 1 = water, 2 = oil)"
        )
    return np.ascontiguousarray(arr, dtype=np.uint8)


def load_volume(path: str | Path) -> np.ndarray:
    """Load a segmented volume from .npy and return a uint8 array (z, y, x).

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    ValueError
        The file cannot be read as a numeric .npy or fails ``validate_volume``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    try:
        raw = np.load(path, allow_pickle=False)
    except ValueError as exc:  # object array or corrupted file
        raise ValueError(f"{path}: cannot read as a numeric .npy: {exc}") from exc
    if isinstance(raw, np.lib.npyio.NpzFile):
        raw.close()
        # wrong file format rather than wrong argument type, hence ValueError
        raise ValueError(f"{path}: expected a .npy with one array, not an .npz archive")  # noqa: TRY004
    try:
        return validate_volume(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def phase_fractions(volume: np.ndarray) -> dict[int, float]:
    """Phase fractions of the volume: {label: fraction}."""
    counts = np.bincount(np.asarray(volume, dtype=np.uint8).ravel(), minlength=len(ALL_PHASES))
    total = counts.sum()
    return {label: float(counts[label] / total) for label in ALL_PHASES}


def has_oil(volume: np.ndarray) -> bool:
    """Whether the volume has at least one oil voxel."""
    return bool(np.any(volume == OIL))


# --------------------------------------------------------------------------- droplets.csv


def _format_cell(column: str, value: Any) -> str:
    if value is None:
        return ""
    kind = DROPLET_COLUMNS[column]
    if kind is bool:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"column {column!r}: expected bool, got {value!r}")
        return "true" if value else "false"
    if kind is int:
        if isinstance(value, (bool, np.bool_)) or int(value) != value:
            raise TypeError(f"column {column!r}: expected an integer, got {value!r}")
        return str(int(value))
    if kind is float:
        return repr(float(value))
    if column == "status" and value not in STATUSES:
        raise ValueError(f"unknown status {value!r}; allowed {sorted(STATUSES)}")
    if column == "phase" and value not in DROPLET_PHASES:
        raise ValueError(f"unknown droplet phase {value!r}; allowed {DROPLET_PHASES}")
    return str(value)


def _parse_cell(column: str, text: str) -> Any:
    if text == "":
        return None
    kind = DROPLET_COLUMNS[column]
    if kind is bool:
        if text not in ("true", "false"):
            raise ValueError(f"column {column!r}: expected true/false, got {text!r}")
        return text == "true"
    return kind(text)


def write_droplets_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the per-droplet table.

    Each row is a dict with keys from ``DROPLET_COLUMNS``. Missing keys and ``None`` give an
    empty cell (e.g. ``theta_macro_deg`` of a droplet whose status is not ok).
    Extra keys are an error, so that typos in names are not silently lost.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(DROPLET_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, row in enumerate(rows):
            unknown = set(row) - set(columns)
            if unknown:
                raise KeyError(f"row {i}: unknown columns {sorted(unknown)}")
            writer.writerow([_format_cell(c, row.get(c)) for c in columns])
    return path


def read_droplets_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read ``droplets.csv`` back with the types from ``DROPLET_COLUMNS``."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(DROPLET_COLUMNS):
            raise ValueError(f"{path}: header does not match DROPLET_COLUMNS")
        return [{c: _parse_cell(c, row[c]) for c in DROPLET_COLUMNS} for row in reader]


# --------------------------------------------------------------------------- summary.json


def summarize_droplets(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summary statistics of theta_macro over droplets with status ok.

    The standard deviation is the sample one (ddof=1); None for fewer than 2 droplets.
    """
    rows = list(rows)
    thetas = np.array(
        [
            r["theta_macro_deg"]
            for r in rows
            if r.get("status") == STATUS_OK and r.get("theta_macro_deg") is not None
        ],
        dtype=np.float64,
    )
    n = int(thetas.size)
    status_counts = {s: sum(1 for r in rows if r.get("status") == s) for s in sorted(STATUSES)}
    return {
        "n_droplets_total": len(rows),
        "n_droplets_ok": n,
        "status_counts": status_counts,
        "theta_mean_deg": float(thetas.mean()) if n else None,
        "theta_std_deg": float(thetas.std(ddof=1)) if n >= 2 else None,
        "theta_median_deg": float(np.median(thetas)) if n else None,
    }


def _to_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_json(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> Path:
    """Write the summary as JSON (UTF-8, indented). numpy types are converted to Python,
    NaN/inf are written as null."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_json(summary), ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def read_summary_json(path: str | Path) -> dict[str, Any]:
    """Read ``summary.json``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
