"""Synthetic reference volumes with a known answer.

A spherical oil droplet on a flat substrate (Eq. 12 of the paper).

Conventions:
- array axes (z, y, x), labels 0 = solid, 1 = water, 2 = oil;
- lengths in voxels, angles in radians;
- the contact angle ``theta`` is measured **through water**: theta_w = pi - theta_d,
  where theta_d is the angle through the droplet.

Geometry. The substrate is the half-space z < z_plane. The droplet is the part of a ball of
radius R above the plane; the ball centre lies R*cos(theta_w) above the plane.
A voxel belongs to a phase according to its centre (integer coordinates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gauss_bonnet.phases import OIL, SOLID, WATER


@dataclass(frozen=True)
class SphericalCap:
    """Spherical droplet on a plane: labelled volume and analytic quantities."""

    volume: np.ndarray  # (nz, ny, nx), uint8, phase labels
    radius: float  # R, vox
    theta: float  # theta_w, contact angle through water, rad
    z_plane: float  # z coordinate of the substrate plane, vox
    center: tuple[float, float, float]  # ball centre (z, y, x), vox

    @property
    def height(self) -> float:
        """h, droplet height above the substrate, vox: h = R(1 + cos theta_w)."""
        return self.radius * (1.0 + math.cos(self.theta))

    @property
    def base_radius(self) -> float:
        """a, contact spot radius, vox: a = R sin theta_w."""
        return self.radius * math.sin(self.theta)

    @property
    def oil_volume(self) -> float:
        """Spherical cap volume, vox^3: V = pi h^2 (3R - h) / 3."""
        h = self.height
        return math.pi * h * h * (3.0 * self.radius - h) / 3.0

    @property
    def area_lv(self) -> float:
        """A_lv, oil-water interface area, vox^2: 2 pi R h (Eq. 12)."""
        return 2.0 * math.pi * self.radius * self.height

    @property
    def area_ls(self) -> float:
        """A_ls, oil-solid interface area, vox^2: pi a^2 = pi h (2R - h) (Eq. 12)."""
        return math.pi * self.base_radius**2

    @property
    def contact_line_length(self) -> float:
        """Contact line length, vox: 2 pi a."""
        return 2.0 * math.pi * self.base_radius

    @property
    def n_loops(self) -> int:
        """N, number of contact line loops."""
        return 1

    @property
    def euler_surface(self) -> int:
        """Euler characteristic of the droplet surface (a topological sphere)."""
        return 2

    @property
    def sum_kappa_g_area_lv(self) -> float:
        """Integral of kappa_G over I_lv = A_lv / R^2 = 2 pi (1 + cos theta_w)."""
        return 2.0 * math.pi * (1.0 + math.cos(self.theta))

    @property
    def sum_kappa_g_area_ls(self) -> float:
        """Integral of kappa_G over I_ls = 0 (flat spot)."""
        return 0.0

    @property
    def deficit_curvature(self) -> float:
        """Deficit curvature 4 pi - sum(kappa_G A) = 2 pi (1 - cos theta_w), rad (Eq. 4, 5)."""
        return 4.0 * math.pi - self.sum_kappa_g_area_lv - self.sum_kappa_g_area_ls

    @property
    def theta_macro(self) -> float:
        """Expected theta_macro by Eq. 20: deficit / (4N) = pi (1 - cos theta_w) / 2, rad."""
        return self.deficit_curvature / (4.0 * self.n_loops)


def expected_theta_macro(theta: float) -> float:
    """theta_macro (Eq. 20) of a smooth spherical droplet with angle theta_w through water, rad.

    The map theta_w -> theta_macro is monotonic but nonlinear: it equals theta_w only
    at 0, pi/2 and pi.
    """
    return math.pi * (1.0 - math.cos(theta)) / 2.0


def spherical_cap(
    radius: float,
    theta: float,
    *,
    solid_thickness: int = 3,
    margin: int = 3,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SphericalCap:
    """Build a volume with a spherical oil droplet on a flat substrate.

    Parameters
    ----------
    radius
        R, droplet radius of curvature, vox (> 0).
    theta
        theta_w, contact angle through water, rad, in (0, pi).
        Small theta_w gives an almost full ball (oil is non-wetting), large a spread lens.
    solid_thickness
        Number of solid voxel layers under the droplet (>= 1).
    margin
        Minimum water gap between the droplet and the volume boundary, vox (>= 1),
        so that the droplet does not touch the image edge.
    offset
        Sub-voxel shift (dz, dy, dx) of the ball centre and the plane, each in [0, 1).
        Used to check that the result does not depend on the voxel grid position.

    Raises
    ------
    ValueError
        On invalid parameters or if the grid does not resolve the droplet
        (height h < 1 or spot radius a < 1 voxel).
    """
    if not radius > 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if not 0.0 < theta < math.pi:
        raise ValueError(f"theta must be in (0, pi) rad, got {theta}")
    if solid_thickness < 1:
        raise ValueError(f"solid_thickness must be >= 1, got {solid_thickness}")
    if margin < 1:
        raise ValueError(f"margin must be >= 1, got {margin}")
    if len(offset) != 3 or not all(0.0 <= o < 1.0 for o in offset):
        raise ValueError(f"offset must be three numbers in [0, 1), got {offset}")

    cos_t = math.cos(theta)
    height = radius * (1.0 + cos_t)
    base_radius = radius * math.sin(theta)
    if height < 1.0 or base_radius < 1.0:
        raise ValueError(
            f"the grid does not resolve the droplet: h = {height:.3f}, a = {base_radius:.3f} vox "
            "(need >= 1); increase radius or move theta away from 0 and pi"
        )

    # Horizontal half-width of the droplet: R if the ball equator is above the plane, else a.
    half_width = radius if cos_t >= 0.0 else base_radius

    dz, dy, dx = offset
    # Solid voxel centres: z = 0 ... solid_thickness - 1.
    z_plane = solid_thickness - 0.5 + dz
    nz = solid_thickness + math.ceil(height + dz) + margin + 1
    nyx = 2 * (math.ceil(half_width) + margin + 1) + 1
    cy = (nyx - 1) / 2.0 + dy
    cx = (nyx - 1) / 2.0 + dx
    cz = z_plane + radius * cos_t

    z = np.arange(nz, dtype=np.float64)[:, None, None]
    y = np.arange(nyx, dtype=np.float64)[None, :, None]
    x = np.arange(nyx, dtype=np.float64)[None, None, :]

    in_ball = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius * radius
    below_plane = np.broadcast_to(z < z_plane, in_ball.shape)

    volume = np.full(in_ball.shape, WATER, dtype=np.uint8)
    volume[in_ball & ~below_plane] = OIL
    volume[below_plane] = SOLID

    return SphericalCap(
        volume=volume,
        radius=float(radius),
        theta=float(theta),
        z_plane=float(z_plane),
        center=(float(cz), float(cy), float(cx)),
    )
