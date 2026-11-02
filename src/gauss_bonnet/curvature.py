"""Per-triangle curvature of the droplet surface: quadric fit, kappa_G, kappa_M (Eq. 19, Algorithm 1).

For each triangle i:

1. **Neighbourhood**: vertices within ``n_layer`` rings of the triangle vertices, where rings
   grow only along edges of faces of **the same region** (I_lv or I_ls). The fit never crosses
   the contact line; otherwise the kink along the line would be smeared into sum(kappa_G A)
   and theta_macro would drift towards 0.
2. **Local frame**: the origin is the triangle centre, the h axis is the area-weighted normal of
   the first-ring faces pointing out of the droplet, u and v lie in the tangent plane.
3. **Least-squares fit**: h = a u^2 + b uv + c v^2 + d u + e v + g.
   The linear terms absorb the tilt of the frame relative to the true normal.
4. **Curvatures** of the graph at (0, 0): first and second fundamental forms
   I = [[1 + d^2, de], [de, 1 + e^2]], II = -[[2a, b], [b, 2c]] / sqrt(1 + d^2 + e^2).
   kappa_G = det II / det I, kappa_M = tr(I^-1 II) / 2,
   kappa_1,2 = kappa_M +- sqrt(kappa_M^2 - kappa_G) (Eq. 19).

Sign: the normal points out of the droplet, so **kappa_M > 0 on a convex patch** (a ball of
radius R: kappa_M = 1/R, kappa_G = 1/R^2), as in Fig. 3 of the paper. The sign correction of
kappa_G A (Algorithm 1, step 5) is done in ``contact_angle.py``.

If the neighbourhood has fewer than ``MIN_FIT_POINTS`` vertices or the system is degenerate,
the triangle curvature is set to 0 (flat patch) and ``fit_ok = False``.

**Neighbourhood size.** A "layer" here is a ring of mesh vertices (~0.7 vox). A height-field
quadric fit overestimates curvature when the neighbourhood is comparable to the radius of
curvature: on a ball with R = 4, sum(kappa_G A) = 1.26 * 4 pi at n_layer = 2 and 4.7 * 4 pi at
n_layer = 7; at R = 20 it is 0.99...1.07 * 4 pi. Hence the default n_layer = 3 rather than 10 as
in Fig. 3 (where a "layer" is defined differently).

**Method ``"angle_defect"``** (not from the paper). kappa_G A is taken from the discrete
Gaussian curvature of the mesh, the angle defect 2 pi - sum of angles at interior vertices of
a region. A vertex defect is split equally between its faces. With it Gauss-Bonnet holds
exactly on the mesh, and the result barely depends on the droplet size. Contact line vertices
are excluded from the sum: their defect is the kink itself, i.e. part of the deficit
curvature. In this method kappa_M still comes from the quadric fit (needed for the sign
correction, Algorithm 1); kappa_1,2 = kappa_M +- sqrt(max(kappa_M^2 - kappa_G, 0)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from gauss_bonnet.mesh import ROLE_LS, ROLE_LV, DropletMesh, _directed_edges
from gauss_bonnet.phases import SOLID, WATER

DEFAULT_N_LAYER = 3  # vertex rings; the paper (Fig. 3) uses 5, 10, 15 with a different layer definition
CURVATURE_METHODS = ("quadric", "angle_defect")
DEFAULT_METHOD = "angle_defect"  # "quadric" follows the paper
MIN_FIT_POINTS = 9  # 6 unknowns + margin
_CHUNK_PAIRS = 400_000  # (triangle, vertex) pairs per vectorised step


@dataclass(frozen=True)
class FaceCurvature:
    """Curvatures on mesh triangles (arrays of length F), vox^-1 and vox^-2."""

    kappa1: np.ndarray  # kappa_1 >= kappa_2
    kappa2: np.ndarray
    kappa_g: np.ndarray  # kappa_G = kappa_1 kappa_2
    kappa_m: np.ndarray  # kappa_M = (kappa_1 + kappa_2) / 2
    n_points: np.ndarray  # number of vertices in the fit
    fit_ok: np.ndarray  # bool

    def sum_kappa_g_area(self, mesh: DropletMesh, region: int | None = None) -> float:
        """Sum of kappa_G A over a region (WATER = I_lv, SOLID = I_ls, None = whole surface), no sign correction."""
        mask = np.ones(len(mesh.faces), dtype=bool) if region is None else mesh.face_phase == region
        return float(np.sum(self.kappa_g[mask] * mesh.face_areas[mask]))


def face_curvature(
    mesh: DropletMesh,
    n_layer: int = DEFAULT_N_LAYER,
    *,
    method: str = DEFAULT_METHOD,
) -> FaceCurvature:
    """Principal, Gaussian and mean curvatures on each triangle (Eq. 19).

    ``method="quadric"``: quadric fit (Algorithm 1); ``"angle_defect"``: kappa_G from the mesh
    angle defect, kappa_M from the quadric fit.
    """
    if n_layer < 1:
        raise ValueError(f"n_layer must be >= 1, got {n_layer}")
    if method not in CURVATURE_METHODS:
        raise ValueError(f"method must be one of {CURVATURE_METHODS}, got {method!r}")
    n_faces = len(mesh.faces)
    k1 = np.zeros(n_faces)
    k2 = np.zeros(n_faces)
    kg = np.zeros(n_faces)
    km = np.zeros(n_faces)
    n_points = np.zeros(n_faces, dtype=np.int64)
    fit_ok = np.zeros(n_faces, dtype=bool)

    for region in (WATER, SOLID):
        idx = np.flatnonzero(mesh.face_phase == region)
        if len(idx) == 0:
            continue
        hood, ring1 = _neighbourhoods(mesh, idx, n_layer)
        frame_n = _ring_normals(mesh, idx, ring1)
        res = _fit_quadrics(mesh, idx, hood, frame_n)
        k1[idx], k2[idx], kg[idx], km[idx], n_points[idx], fit_ok[idx] = res

    if method == "angle_defect":
        kg = face_angle_defect_curvature(mesh)
        disc = np.sqrt(np.maximum(km * km - kg, 0.0))
        k1, k2 = km + disc, km - disc

    return FaceCurvature(k1, k2, kg, km, n_points, fit_ok)


# --------------------------------------------------------------------------- angle defect


def vertex_angle_defects(mesh: DropletMesh) -> np.ndarray:
    """Angle defect 2 pi - sum of triangle angles at each vertex, rad."""
    v = mesh.verts[mesh.faces]
    angles = np.zeros(len(mesh.verts))
    for k in range(3):
        a = v[:, (k + 1) % 3] - v[:, k]
        b = v[:, (k + 2) % 3] - v[:, k]
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-300)
        ang = np.where((na > 0) & (nb > 0), np.arccos(np.clip(cos, -1.0, 1.0)), 0.0)
        np.add.at(angles, mesh.faces[:, k], ang)
    return 2 * np.pi - angles


def face_angle_defect_curvature(mesh: DropletMesh) -> np.ndarray:
    """kappa_G on faces from the angle defect of interior I_lv and I_ls vertices, vox^-2.

    sum_f kappa_G,f A_f equals the sum of interior vertex defects of a region (exactly).
    """
    interior = np.isin(mesh.vertex_roles, (ROLE_LV, ROLE_LS))
    defect = np.where(interior, vertex_angle_defects(mesh), 0.0)
    vert_area = np.zeros(len(mesh.verts))
    np.add.at(vert_area, mesh.faces.ravel(), np.repeat(mesh.face_areas / 3.0, 3))
    kv = np.divide(defect, vert_area, out=np.zeros_like(defect), where=vert_area > 0)
    return kv[mesh.faces].mean(axis=1)


# --------------------------------------------------------------------------- neighbourhoods


def _neighbourhoods(mesh: DropletMesh, idx: np.ndarray, n_layer: int):
    """Neighbourhoods of triangles ``idx`` within their region.

    Returns
    -------
    hood : csr (len(idx) x V), vertices within ``n_layer`` rings;
    ring1 : csr (len(idx) x len(idx)), region faces sharing a vertex with the triangle.
    """
    n = len(mesh.verts)
    faces = mesh.faces[idx]
    m = len(idx)
    rows = np.repeat(np.arange(m), 3)
    incidence = sparse.csr_matrix((np.ones(3 * m), (rows, faces.ravel())), shape=(m, n))

    e = np.unique(np.sort(_directed_edges(faces), axis=1), axis=0)
    adj = sparse.csr_matrix(
        (np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)
    )
    step = (adj + sparse.identity(n, format="csr")).tocsr()
    step.data[:] = 1.0

    hood = incidence.copy()
    for _ in range(n_layer):
        hood = hood @ step
        hood.data[:] = 1.0
    hood.sort_indices()

    ring1 = incidence @ incidence.T
    ring1.data[:] = 1.0
    return hood.tocsr(), ring1.tocsr()


def _ring_normals(mesh: DropletMesh, idx: np.ndarray, ring1: sparse.csr_matrix) -> np.ndarray:
    """Area-weighted normal of the first-ring faces (unit, outward)."""
    weighted = mesh.face_normals[idx] * mesh.face_areas[idx, None]
    n = ring1 @ weighted
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    bad = norm[:, 0] < 1e-12
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    n[bad] = mesh.face_normals[idx][bad]
    return n


# --------------------------------------------------------------------------- fitting


def _tangent_frames(n: np.ndarray):
    """Orthonormal t1, t2 perpendicular to n."""
    helper = np.zeros_like(n)
    axis = np.argmin(np.abs(n), axis=1)
    helper[np.arange(len(n)), axis] = 1.0
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    return t1, t2


def _fit_quadrics(mesh: DropletMesh, idx, hood: sparse.csr_matrix, frame_n: np.ndarray):
    m = len(idx)
    centres = mesh.verts[mesh.faces[idx]].mean(axis=1)
    t1, t2 = _tangent_frames(frame_n)
    counts = np.diff(hood.indptr)

    coef = np.zeros((m, 6))  # a, b, c, d, e, g in scaled coordinates
    scale = np.ones(m)
    ok = counts >= MIN_FIT_POINTS

    start = 0
    while start < m:
        # take rows until the number of pairs exceeds the limit
        stop = int(np.searchsorted(hood.indptr, hood.indptr[start] + _CHUNK_PAIRS, side="right"))
        stop = min(max(stop - 1, start + 1), m)
        rows = np.repeat(np.arange(start, stop), counts[start:stop])
        cols = hood.indices[hood.indptr[start] : hood.indptr[stop]]

        p = mesh.verts[cols] - centres[rows]
        u = np.einsum("ij,ij->i", p, t1[rows])
        v = np.einsum("ij,ij->i", p, t2[rows])
        h = np.einsum("ij,ij->i", p, frame_n[rows])

        # scale is the RMS tangential distance, for conditioning
        r2 = np.bincount(rows - start, weights=u * u + v * v, minlength=stop - start)
        s = np.sqrt(r2 / np.maximum(counts[start:stop], 1))
        s[s < 1e-12] = 1.0
        scale[start:stop] = s
        sr = s[rows - start]
        u, v, h = u / sr, v / sr, h / sr

        phi = np.stack([u * u, u * v, v * v, u, v, np.ones_like(u)], axis=1)
        offsets = hood.indptr[start:stop] - hood.indptr[start]
        nonempty = counts[start:stop] > 0
        gram = np.zeros((stop - start, 6, 6))
        rhs = np.zeros((stop - start, 6))
        if np.any(nonempty):
            gram[nonempty] = np.add.reduceat(
                phi[:, :, None] * phi[:, None, :], offsets[nonempty], axis=0
            )
            rhs[nonempty] = np.add.reduceat(phi * h[:, None], offsets[nonempty], axis=0)

        chunk_ok = ok[start:stop].copy()
        if np.any(chunk_ok):
            g = gram[chunk_ok]
            cond = np.linalg.cond(g)
            good = cond < 1e10
            sol = np.zeros((len(g), 6))
            if np.any(good):
                sol[good] = np.linalg.solve(g[good], rhs[chunk_ok][good][:, :, None])[:, :, 0]
            sub = np.flatnonzero(chunk_ok)
            chunk_ok[sub[~good]] = False
            coef[start:stop][chunk_ok] = sol[good]
        ok[start:stop] = chunk_ok
        start = stop

    a, b, c, d, e = (coef[:, k] for k in range(5))
    fuu, fuv, fvv = 2 * a / scale, b / scale, 2 * c / scale  # second derivatives in vox^-1
    w2 = 1.0 + d * d + e * e
    # the outward normal is +h and kappa_M > 0 on a convex surface, hence the minus sign
    kg = (fuu * fvv - fuv * fuv) / (w2 * w2)
    km = -((1 + e * e) * fuu - 2 * d * e * fuv + (1 + d * d) * fvv) / (2 * w2**1.5)
    disc = np.sqrt(np.maximum(km * km - kg, 0.0))
    k1, k2 = km + disc, km - disc

    for arr in (k1, k2, kg, km):
        arr[~ok] = 0.0
    return k1, k2, kg, km, counts, ok
