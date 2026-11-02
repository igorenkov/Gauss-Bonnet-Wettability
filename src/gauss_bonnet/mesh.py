"""Triangulated droplet surface: marching cubes, lv/ls labelling, smoothing (Eq. 17-18).

Steps (Algorithm 1):

1. ``build_mesh``: marching cubes (Lewiner, ``skimage.measure.marching_cubes``) on the binary
   oil mask of a ``DropletCrop`` at level ``MC_LEVEL`` = 0.5 - 1e-3, after which the vertices
   are snapped back to edge midpoints. For binary data every vertex lies at the midpoint of an
   edge between an oil voxel and a voxel of another phase, so the phase outside the vertex
   (water or solid) is known exactly.

   **Why not exactly 0.5.** On an ambiguous cube face (two oil voxels on a diagonal) the value
   at the saddle point is 0.5, i.e. equal to the level. Lewiner resolves this tie
   inconsistently, and thin wedges near the wall get false tunnels (Euler characteristic -62
   instead of 2 on a capillary meniscus). A level slightly below 0.5 always connects diagonal
   oil voxels, consistent with 26-connectivity of droplets.

   A triangle belongs to I_ls only if **all three** of its vertices have solid outside,
   otherwise to I_lv. Thus the contact spot consists of faces lying on the solid surface, and
   the contact line (the spot boundary) lies entirely on the solid. Step-like triangles at the
   spot edge belong to I_lv, which is what meets the solid. A majority rule gave a contact line
   that zigzags in height. Faces are oriented with normals pointing out of the droplet.

   **Triple line correction** (``triple_line=True``). In binary MC the contact line passes
   through the centres of the outermost oil voxels, and the first I_lv vertex lies half a voxel
   higher and further out. The surface overhangs at the edge, and the mesh meets the substrate
   at about 90 degrees for any true angle. The missing turn goes into the I_lv curvature and
   distorts theta_macro (a 135 degree reference gives about 35 degrees without the correction).
   A multi-label MC (ref. [43] of the paper) would put the triple line on the edge shared by
   oil, water and two solid voxels. The correction does the same: a vertex on an
   "oil o - water w" edge whose neighbours o + s*e_j and w + s*e_j along one axis j
   perpendicular to the edge are both solid is shifted by s*e_j/2 towards the solid and treated
   as lying on the solid. If there is solid on both sides along axis j, the vertex is not
   shifted along that axis.
2. ``smooth_mesh``: vertex smoothing (Eq. 17-18), **separately per region**: interior I_lv
   vertices are averaged over I_lv neighbours only, I_ls vertices over I_ls, contact line
   vertices over contact line neighbours only. Taubin by default (an alpha step alternates
   with a mu < 0 step) so that the droplet does not shrink; "laplacian" is literally Eq. 18.

Why separately. sum(kappa_G A) over a closed smooth surface is 4 pi chi, and then the deficit
curvature is 0 (Eq. 4). A non-zero deficit curvature is curvature concentrated on the surface
kink along the contact line (Eq. 2, 3). Smoothing across the contact line rounds the kink,
moves its curvature into the smooth parts and underestimates theta_macro.

Mesh simplification by edge collapse is not implemented.

Vertex coordinates are global voxel indices (z, y, x) of the source volume.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
from scipy import sparse
from skimage import measure

from gauss_bonnet.droplets import DropletCrop
from gauss_bonnet.phases import OIL, SOLID, WATER

SMOOTH_METHODS = ("laplacian", "taubin")
DEFAULT_ALPHA = 0.5
DEFAULT_ITERATIONS = 10
TAUBIN_PASSBAND = 0.1
DEFAULT_METHOD = "taubin"  # no shrinkage; "laplacian" is literally Eq. 17-18
MC_LEVEL = 0.5 - 1e-3  # just below 0.5: removes ties at face saddle points (see module docstring)

# Vertex roles during smoothing
ROLE_LV = 0  # interior I_lv vertex
ROLE_LS = 1  # interior I_ls vertex
ROLE_CONTACT = 2  # contact line vertex
ROLE_FIXED = 3  # singular vertex (regions touch at a point only), kept fixed


@dataclass(frozen=True)
class DropletMesh:
    """Closed surface of one droplet."""

    verts: np.ndarray  # (V, 3) float64, global coordinates (z, y, x), vox
    faces: np.ndarray  # (F, 3) int64, oriented with outward normals
    face_phase: np.ndarray  # (F,) uint8: WATER = I_lv, SOLID = I_ls
    droplet_id: int = 0

    # ---------------------------------------------------------------- areas, normals

    @cached_property
    def face_areas(self) -> np.ndarray:
        """A_i = |v12 x v13| / 2 (Algorithm 1), vox^2."""
        return 0.5 * np.linalg.norm(self._face_cross, axis=1)

    @cached_property
    def face_normals(self) -> np.ndarray:
        """Unit face normals, pointing out of the droplet."""
        cross = self._face_cross
        norm = np.linalg.norm(cross, axis=1, keepdims=True)
        return np.divide(cross, norm, out=np.zeros_like(cross), where=norm > 0)

    @cached_property
    def _face_cross(self) -> np.ndarray:
        v = self.verts[self.faces]
        return np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])

    @property
    def is_lv(self) -> np.ndarray:
        return self.face_phase == WATER

    @property
    def is_ls(self) -> np.ndarray:
        return self.face_phase == SOLID

    @property
    def area_lv(self) -> float:
        return float(self.face_areas[self.is_lv].sum())

    @property
    def area_ls(self) -> float:
        return float(self.face_areas[self.is_ls].sum())

    @property
    def enclosed_volume(self) -> float:
        """Volume enclosed by the mesh (divergence theorem), vox^3. > 0 for outward normals."""
        return _signed_volume(self.verts, self.faces)

    # ---------------------------------------------------------------- edges and contact line

    @cached_property
    def edges(self) -> np.ndarray:
        """Unique undirected edges (E, 2), i < j."""
        return np.unique(np.sort(_directed_edges(self.faces), axis=1), axis=0)

    @cached_property
    def edge_faces(self) -> np.ndarray:
        """The two adjacent faces of each edge in ``edges`` (E, 2); -1 if there is only one."""
        directed = _directed_edges(self.faces)
        keys = np.sort(directed, axis=1)
        face_idx = np.repeat(np.arange(len(self.faces)), 3)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys, face_idx = keys[order], face_idx[order]
        _, start, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
        out = np.full((len(start), 2), -1, dtype=np.int64)
        out[:, 0] = face_idx[start]
        two = counts >= 2
        out[two, 1] = face_idx[start[two] + 1]
        return out

    @cached_property
    def contact_edges(self) -> np.ndarray:
        """Contact line edges: between an I_lv face and an I_ls face (K, 2)."""
        ef = self.edge_faces
        valid = ef[:, 1] >= 0
        p0 = self.face_phase[ef[:, 0]]
        p1 = np.where(valid, self.face_phase[np.maximum(ef[:, 1], 0)], p0)
        return self.edges[valid & (p0 != p1)]

    @cached_property
    def vertex_roles(self) -> np.ndarray:
        """Role of each vertex during smoothing (ROLE_*)."""
        n = len(self.verts)
        touches_lv = np.zeros(n, dtype=bool)
        touches_ls = np.zeros(n, dtype=bool)
        touches_lv[self.faces[self.is_lv].ravel()] = True
        touches_ls[self.faces[self.is_ls].ravel()] = True
        on_line = np.zeros(n, dtype=bool)
        on_line[self.contact_edges.ravel()] = True

        roles = np.full(n, ROLE_FIXED, dtype=np.int8)
        roles[touches_lv & ~touches_ls] = ROLE_LV
        roles[touches_ls & ~touches_lv] = ROLE_LS
        roles[touches_lv & touches_ls & on_line] = ROLE_CONTACT
        return roles

    def is_closed_manifold(self) -> bool:
        """Every edge belongs to exactly two faces with opposite orientation."""
        directed = _directed_edges(self.faces)
        _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
        if not np.all(counts == 2):
            return False
        # consistent orientation: each directed edge occurs once
        return len(np.unique(directed, axis=0)) == len(directed)


# --------------------------------------------------------------------------- construction


def build_mesh(crop: DropletCrop, *, triple_line: bool = True) -> DropletMesh:
    """Build the closed droplet surface from a crop (no smoothing).

    ``triple_line`` enables the triple line correction (see the module docstring).
    """
    local = crop.volume
    mask = (local == OIL).astype(np.float32)
    if not mask.any():
        raise ValueError(f"droplet {crop.droplet.id}: no oil in the crop")
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any() \
            or mask[:, :, 0].any() or mask[:, :, -1].any():
        raise ValueError(f"droplet {crop.droplet.id}: oil touches the crop edge, pad >= 1 is required")

    verts, faces, _, _ = measure.marching_cubes(
        mask, level=MC_LEVEL, method="lewiner", allow_degenerate=False
    )
    # vertices are 1e-3 off the edge midpoint; snap back to the midpoint (or cube centre)
    verts = np.round(verts.astype(np.float64) * 2) / 2
    faces = faces.astype(np.int64)

    oil_vox, out_vox, axis, center = _vertex_edges(local, verts)
    vert_phase = local[tuple(out_vox.T)].astype(np.uint8)
    vert_phase[center] = _center_phase(local, verts[center])
    if triple_line:
        shift = _triple_line_shift(local, oil_vox, out_vox, axis, vert_phase)
        shift[center] = 0.0  # a cube centre is not on an edge, nothing to shift
        moved = np.any(shift != 0, axis=1)
        verts = verts + shift
        vert_phase[moved] = SOLID  # the vertex now lies on the solid surface
    all_solid = np.all(vert_phase[faces] == SOLID, axis=1)
    face_phase = np.where(all_solid, SOLID, WATER).astype(np.uint8)

    if _signed_volume(verts, faces) < 0:
        faces = faces[:, ::-1].copy()

    verts += np.asarray(crop.offset, dtype=np.float64)
    return DropletMesh(verts=verts, faces=faces, face_phase=face_phase, droplet_id=crop.droplet.id)


_CUBE_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def _vertex_edges(local: np.ndarray, verts: np.ndarray):
    """For each MC vertex: the oil voxel, the other-phase voxel on its edge, and the edge axis.

    In ambiguous cubes (MC33) Lewiner may add a vertex at the cube centre: all three of its
    coordinates are half-integers and there is no edge. At ``MC_LEVEL`` this has not been
    observed (such vertices came from ties at level 0.5), but the handling is kept. Such
    vertices are flagged in ``center``; for them ``oil_vox``/``out_vox`` are arbitrary cube
    corners and the phase is given by ``_center_phase``.
    """
    n = len(verts)
    lo = np.floor(verts).astype(np.int64)
    frac = verts - lo
    half = np.abs(frac - 0.5) < 1e-6
    whole = np.abs(frac - np.round(frac)) < 1e-6
    n_half = half.sum(axis=1)
    center = n_half == 3
    on_edge = (n_half == 1) & np.all(half | whole, axis=1)
    if not np.all(on_edge | center):
        raise RuntimeError("MC vertices are neither at edge midpoints nor at cube centres; is the input binary?")
    axis = np.where(center, 0, np.argmax(half, axis=1))
    hi = lo.copy()
    hi[np.arange(n), axis] += 1
    lo_is_oil = local[tuple(lo.T)] == OIL
    crosses = lo_is_oil != (local[tuple(hi.T)] == OIL)
    if np.any(~crosses & ~center):
        raise RuntimeError("the edge of an MC vertex does not cross the oil boundary")
    oil_vox = np.where(lo_is_oil[:, None], lo, hi)
    out_vox = np.where(lo_is_oil[:, None], hi, lo)
    return oil_vox, out_vox, axis, center


def _center_phase(local: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Outside phase of a cube-centre vertex: SOLID if all non-oil corners are solid."""
    if len(verts) == 0:
        return np.zeros(0, dtype=np.uint8)
    corners = np.floor(verts).astype(np.int64)[:, None, :] + _CUBE_CORNERS[None]
    phases = local[tuple(np.moveaxis(corners, -1, 0))]  # (V, 8)
    outside = phases != OIL
    all_solid = np.all(~outside | (phases == SOLID), axis=1)
    return np.where(all_solid, SOLID, WATER).astype(np.uint8)


def _triple_line_shift(local, oil_vox, out_vox, axis, vert_phase) -> np.ndarray:
    """Shift of oil-water vertices onto the triple line (see the module docstring), vox."""
    shape = np.asarray(local.shape)
    shift = np.zeros(oil_vox.shape, dtype=np.float64)
    is_water = vert_phase == WATER
    for j in range(3):
        hits = []
        for s in (-1, 1):
            o, w = oil_vox.copy(), out_vox.copy()
            o[:, j] += s
            w[:, j] += s
            inside = np.all((o >= 0) & (o < shape) & (w >= 0) & (w < shape), axis=1)
            o, w = np.clip(o, 0, shape - 1), np.clip(w, 0, shape - 1)
            hits.append(
                is_water & (axis != j) & inside
                & (local[tuple(o.T)] == SOLID) & (local[tuple(w.T)] == SOLID)
            )
        lo_hit, hi_hit = hits
        shift[lo_hit & ~hi_hit, j] = -0.5
        shift[hi_hit & ~lo_hit, j] = 0.5
    return shift


# --------------------------------------------------------------------------- smoothing


def smoothing_operator(mesh: DropletMesh) -> sparse.csr_matrix:
    """Matrix W with equal weights w_ij = 1/|N(i)| (Eq. 17), neighbourhoods restricted by vertex role.

    dv = (W - I) v. Vertices without allowed neighbours (ROLE_FIXED) have W_ii = 1, i.e. dv = 0.
    """
    n = len(mesh.verts)
    roles = mesh.vertex_roles
    rows, cols = [], []

    def add_edges(e: np.ndarray, allowed_roles: tuple[int, ...]) -> None:
        for a, b in ((0, 1), (1, 0)):
            keep = np.isin(roles[e[:, a]], allowed_roles)
            rows.append(e[keep, a])
            cols.append(e[keep, b])

    # Interior vertices of a region: edges of that region's faces.
    for phase, role in ((WATER, ROLE_LV), (SOLID, ROLE_LS)):
        f = mesh.faces[mesh.face_phase == phase]
        e = np.unique(np.sort(_directed_edges(f), axis=1), axis=0)
        add_edges(e, (role,))
    # Contact line vertices: contact line edges only.
    add_edges(mesh.contact_edges, (ROLE_CONTACT,))

    rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj.data[:] = 1.0  # duplicate edges -> 1
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros(n), where=deg > 0)
    w = sparse.diags(inv) @ adj
    # vertices without neighbours stay in place
    return (w + sparse.diags((deg == 0).astype(np.float64))).tocsr()


def smooth_mesh(
    mesh: DropletMesh,
    *,
    alpha: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_ITERATIONS,
    method: str = DEFAULT_METHOD,
) -> DropletMesh:
    """Smooth the vertices: v' = v + alpha * sum w_ij (v_j - v_i) (Eq. 17-18), ``iterations`` times.

    ``method="taubin"`` alternates an alpha step and a mu < 0 step (smoothing without shrinkage).
    Topology and face labels do not change.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if method not in SMOOTH_METHODS:
        raise ValueError(f"method must be one of {SMOOTH_METHODS}, got {method!r}")

    w = smoothing_operator(mesh)
    steps = [alpha]
    if method == "taubin":
        mu = 1.0 / (TAUBIN_PASSBAND - 1.0 / alpha)  # mu < -alpha
        steps.append(mu)

    v = mesh.verts.copy()
    for _ in range(iterations):
        for factor in steps:
            v = v + factor * (w @ v - v)
    return replace(mesh, verts=v)


def mesh_droplet(
    crop: DropletCrop,
    *,
    alpha: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_ITERATIONS,
    method: str = DEFAULT_METHOD,
) -> DropletMesh:
    """build_mesh + smooth_mesh."""
    return smooth_mesh(build_mesh(crop), alpha=alpha, iterations=iterations, method=method)


# --------------------------------------------------------------------------- utilities


def _directed_edges(faces: np.ndarray) -> np.ndarray:
    return faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)


def _signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    v = verts[faces]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)
