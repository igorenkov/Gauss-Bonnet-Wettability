"""Per-droplet result: sign correction, deficit curvature and theta_macro (Eq. 4, 20, Algorithm 1, steps 5-6).

The module neither builds the mesh nor computes curvature: it takes a ready ``DropletMesh``,
``MeshTopology`` and ``FaceCurvature``.

* **Sign correction** (Algorithm 1, step 5), enabled by ``sign_correction=True``: if
  kappa_M < 0 and kappa_G > ``flat_eps``, then kappa_G A -> -kappa_G A. Triangles with
  |kappa_G| <= ``flat_eps`` are treated as flat and give 0. **Off by default**: it breaks the
  Gauss-Bonnet identity and moves the median on the 45 degree test core away from 45
  (47.9 -> 54.3 degrees). The deficit and angle without correction and threshold are always
  returned in the ``*_raw`` fields.
* **Deficit curvature = 4 pi chi(D) - sum(kappa_G A)** (Eq. 4). By default chi(D) comes from
  the mesh (``euler="mesh"``); ``euler="ball"`` is literally Eq. 20 (chi(D) = 1).
* **theta_macro = deficit / (4N)** (Eq. 20).
* **Angle through water**: for water droplets (``phase="water"``) the deficit and angle are
  computed for the water droplet (angle through oil), and droplets.csv gets pi - theta.
* **Statuses:** N = 0 -> ``no_contact`` (theta undefined); theta_macro outside [0, pi] ->
  ``out_of_range`` (the value is not clipped); otherwise ``ok``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from gauss_bonnet.curvature import FaceCurvature
from gauss_bonnet.droplets import Droplet
from gauss_bonnet.io import (
    DROPLET_COLUMNS,
    STATUS_NO_CONTACT,
    STATUS_OK,
    STATUS_OUT_OF_RANGE,
)
from gauss_bonnet.mesh import DropletMesh
from gauss_bonnet.phases import DROPLET_OIL, DROPLET_PHASES, DROPLET_WATER, SOLID, WATER
from gauss_bonnet.topology import MeshTopology

EULER_MODES = ("mesh", "ball")
DEFAULT_EULER = "mesh"
DEFAULT_SIGN_CORRECTION = False  # True follows Algorithm 1
DEFAULT_FLAT_EPS = 0.0  # vox^-2


@dataclass(frozen=True)
class ContactAngle:
    """Result for one droplet. Angles in radians."""

    n_loops: int
    euler_droplet: float  # chi(D) used in Eq. 4
    sum_kappa_g_area_lv: float  # after sign correction
    sum_kappa_g_area_ls: float  # after sign correction
    sum_kappa_g_area_raw: float  # over the whole surface, no correction
    n_flipped: int  # triangles whose sign was flipped
    deficit_curvature: float  # deficit curvature, rad
    deficit_curvature_raw: float
    theta_macro: float | None  # None if N = 0
    theta_macro_raw: float | None
    status: str

    @property
    def theta_macro_deg(self) -> float | None:
        return None if self.theta_macro is None else math.degrees(self.theta_macro)

    @property
    def theta_macro_raw_deg(self) -> float | None:
        return None if self.theta_macro_raw is None else math.degrees(self.theta_macro_raw)


def face_kappa_g_area(
    mesh: DropletMesh,
    curvature: FaceCurvature,
    *,
    sign_correction: bool = DEFAULT_SIGN_CORRECTION,
    flat_eps: float = DEFAULT_FLAT_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """kappa_G A on each triangle after the sign correction (Algorithm 1, step 5).

    Returns
    -------
    values : np.ndarray (F,)
    flipped : np.ndarray (F,) bool, where the sign was flipped.
    """
    if flat_eps < 0:
        raise ValueError(f"flat_eps must be >= 0, got {flat_eps}")
    kg = curvature.kappa_g
    values = kg * mesh.face_areas
    flat = np.abs(kg) <= flat_eps
    if sign_correction:
        flipped = (curvature.kappa_m < 0) & (kg > 0) & ~flat
    else:
        flipped = np.zeros(len(kg), dtype=bool)
    values = np.where(flipped, -values, values)
    values = np.where(flat, 0.0, values)
    return values, flipped


def contact_angle(
    mesh: DropletMesh,
    topology: MeshTopology,
    curvature: FaceCurvature,
    *,
    sign_correction: bool = DEFAULT_SIGN_CORRECTION,
    flat_eps: float = DEFAULT_FLAT_EPS,
    euler: str = DEFAULT_EULER,
) -> ContactAngle:
    """Deficit curvature and theta_macro of a droplet (Eq. 4, 20)."""
    if euler not in EULER_MODES:
        raise ValueError(f"euler must be one of {EULER_MODES}, got {euler!r}")
    chi = topology.euler_droplet if euler == "mesh" else 1.0

    values, flipped = face_kappa_g_area(
        mesh, curvature, sign_correction=sign_correction, flat_eps=flat_eps
    )
    lv = float(values[mesh.face_phase == WATER].sum())
    ls = float(values[mesh.face_phase == SOLID].sum())
    raw = float(np.sum(curvature.kappa_g * mesh.face_areas))

    deficit = 4 * math.pi * chi - (lv + ls)
    deficit_raw = 4 * math.pi * chi - raw

    n = topology.n_loops
    if n == 0:
        theta = theta_raw = None
        status = STATUS_NO_CONTACT
    else:
        theta = deficit / (4 * n)
        theta_raw = deficit_raw / (4 * n)
        status = STATUS_OK if 0.0 <= theta <= math.pi else STATUS_OUT_OF_RANGE

    return ContactAngle(
        n_loops=n,
        euler_droplet=chi,
        sum_kappa_g_area_lv=lv,
        sum_kappa_g_area_ls=ls,
        sum_kappa_g_area_raw=raw,
        n_flipped=int(flipped.sum()),
        deficit_curvature=deficit,
        deficit_curvature_raw=deficit_raw,
        theta_macro=theta,
        theta_macro_raw=theta_raw,
        status=status,
    )


def droplet_row(
    droplet: Droplet,
    mesh: DropletMesh | None = None,
    topology: MeshTopology | None = None,
    angle: ContactAngle | None = None,
    *,
    voxel_size: float = 1.0,
    phase: str = DROPLET_OIL,
) -> dict[str, Any]:
    """A ``droplets.csv`` row (keys from ``io.DROPLET_COLUMNS``).

    ``phase`` is the droplet phase. For a water droplet the angles are converted to the angle
    through water (pi - theta); the deficit and sum(kappa_G A) stay quantities of the droplet
    surface itself.

    For a droplet rejected before meshing (``border``, ``too_small``, voxel-level
    ``no_contact``) only ``droplet`` is passed and the other cells stay empty.
    theta_macro is written for statuses ``ok`` and ``out_of_range``.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    if phase not in DROPLET_PHASES:
        raise ValueError(f"phase must be one of {DROPLET_PHASES}, got {phase!r}")
    row: dict[str, Any] = {
        "id": droplet.id,
        "phase": phase,
        "volume_vox": droplet.volume_vox,
        "centroid_z": droplet.centroid[0],
        "centroid_y": droplet.centroid[1],
        "centroid_x": droplet.centroid[2],
        "touches_border": droplet.touches_border,
        "status": droplet.status,
    }
    if mesh is not None:
        um2 = voxel_size**2
        row.update(
            area_lv=mesh.area_lv,
            area_ls=mesh.area_ls,
            area_lv_um2=mesh.area_lv * um2,
            area_ls_um2=mesh.area_ls * um2,
        )
    if topology is not None:
        row.update(
            n_loops=topology.n_loops,
            n_junctions=topology.n_junctions,
            euler_surface=topology.euler_surface,
            euler_droplet=topology.euler_droplet,
        )
    if angle is not None:
        row.update(
            sum_kappa_g_area_lv=angle.sum_kappa_g_area_lv,
            sum_kappa_g_area_ls=angle.sum_kappa_g_area_ls,
            deficit_curvature=angle.deficit_curvature,
            deficit_curvature_raw=angle.deficit_curvature_raw,
            theta_macro_deg=_through_water(angle.theta_macro_deg, phase),
            theta_macro_raw_deg=_through_water(angle.theta_macro_raw_deg, phase),
            status=angle.status,
        )
    assert set(row) <= set(DROPLET_COLUMNS)
    return row


def _through_water(theta_deg: float | None, phase: str) -> float | None:
    """Angle of a droplet of phase ``phase`` in degrees -> angle through water."""
    if theta_deg is None or phase != DROPLET_WATER:
        return theta_deg
    return 180.0 - theta_deg
