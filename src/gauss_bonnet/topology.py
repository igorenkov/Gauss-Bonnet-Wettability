"""Droplet surface topology: Euler characteristic and contact line loops.

* chi(dD) = V - E + F of the mesh (Algorithm 1). chi(D) = chi(dD) / 2 is an identity for a
  compact 3D body with boundary; a droplet homeomorphic to a ball gives chi(D) = 1.
* N is the number of contact line loops (Eq. 20). Loops are found by walking the boundary of
  the contact spot I_ls along half-edges: from the end of a boundary half-edge the walk turns
  around the vertex over I_ls faces to the next boundary half-edge. Each contact line edge
  belongs to exactly one loop. If two I_ls spots touch at one vertex (a junction, vertex degree
  on the line > 2), the walk gives two separate loops, one per spot.
* A droplet without I_ls faces has N = 0 (``no_contact``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from gauss_bonnet.mesh import DropletMesh
from gauss_bonnet.phases import SOLID, WATER


@dataclass(frozen=True)
class MeshTopology:
    """Topological characteristics of one droplet surface."""

    n_verts: int  # vertices used by at least one face
    n_edges: int
    n_faces: int
    n_components: int  # connected surface components (a cavity inside the droplet adds one)
    loops: tuple[np.ndarray, ...]  # contact line loops: vertex indices in walk order, start not repeated
    n_junctions: int  # vertices where more than two contact line edges meet

    @property
    def euler_surface(self) -> int:
        """chi(dD) = V - E + F."""
        return self.n_verts - self.n_edges + self.n_faces

    @property
    def euler_droplet(self) -> float:
        """chi(D) = chi(dD) / 2."""
        return self.euler_surface / 2

    @property
    def n_loops(self) -> int:
        """N, number of contact line loops."""
        return len(self.loops)

    @property
    def is_ball(self) -> bool:
        """The surface is a single sphere, i.e. chi(D) = 1, as Eq. 20 assumes."""
        return self.n_components == 1 and self.euler_surface == 2


def mesh_topology(mesh: DropletMesh) -> MeshTopology:
    """Compute chi, the number of components and the contact line loops of a droplet mesh."""
    faces = mesh.faces
    used = np.unique(faces)
    n_verts = len(used)
    n_edges = len(mesh.edges)
    n_faces = len(faces)

    # connected components over face edges
    e = mesh.edges
    n = len(mesh.verts)
    adj = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = csgraph.connected_components(adj, directed=False)
    n_components = len(np.unique(comp[used]))

    loops = contact_loops(mesh)

    line = mesh.contact_edges
    if len(line):
        _, deg = np.unique(line.ravel(), return_counts=True)
        n_junctions = int(np.count_nonzero(deg > 2))
    else:
        n_junctions = 0

    return MeshTopology(
        n_verts=n_verts,
        n_edges=n_edges,
        n_faces=n_faces,
        n_components=n_components,
        loops=tuple(loops),
        n_junctions=n_junctions,
    )


def contact_loops(mesh: DropletMesh, region: int = SOLID) -> list[np.ndarray]:
    """Boundary loops of the region ``region`` (SOLID = spot I_ls, WATER = I_lv) on the mesh.

    Both regions share the contact line; they can differ only at junction vertices: walking
    over I_ls splits loops that touch at a point by solid spots, walking over I_lv by I_lv
    regions. I_ls is used by default.

    Returns
    -------
    list of np.ndarray
        Vertex indices of each loop in walk order (oriented like the region faces).
    """
    if region not in (SOLID, WATER):
        raise ValueError(f"region must be SOLID or WATER, got {region}")
    faces = mesh.faces
    n_faces = len(faces)
    if n_faces == 0:
        return []
    n_verts = len(mesh.verts)

    # Half-edge h = 3f + k goes from faces[f, k] to faces[f, (k+1) % 3].
    src = faces.ravel()
    dst = faces[:, [1, 2, 0]].ravel()
    he_face = np.repeat(np.arange(n_faces), 3)

    keys = src * n_verts + dst
    order = np.argsort(keys)
    sorted_keys = keys[order]
    twin_keys = dst * n_verts + src
    pos = np.searchsorted(sorted_keys, twin_keys)
    pos = np.minimum(pos, len(sorted_keys) - 1)
    found = sorted_keys[pos] == twin_keys
    twin = np.where(found, order[pos], -1)

    in_region = mesh.face_phase[he_face] == region
    twin_in_region = np.where(twin >= 0, mesh.face_phase[he_face[np.maximum(twin, 0)]] == region, False)
    boundary = in_region & ~twin_in_region

    def next_in_face(h: int) -> int:
        return h - h % 3 + (h % 3 + 1) % 3

    visited = np.zeros(len(src), dtype=bool)
    loops: list[np.ndarray] = []
    max_turns = len(src)  # guard against infinite loops on a broken mesh
    for start in np.flatnonzero(boundary):
        if visited[start]:
            continue
        verts = []
        h = int(start)
        while not visited[h]:
            visited[h] = True
            verts.append(int(src[h]))
            # turn around the half-edge end over region faces to the next boundary half-edge
            g = next_in_face(h)
            turns = 0
            while not boundary[g]:
                t = int(twin[g])
                if t < 0 or not in_region[t]:  # pragma: no cover - mesh is not closed
                    raise RuntimeError("boundary walk left the region: the mesh is not closed")
                g = next_in_face(t)
                turns += 1
                if turns > max_turns:  # pragma: no cover
                    raise RuntimeError("boundary walk is stuck in a cycle")
            h = g
        if h != start:  # pragma: no cover
            raise RuntimeError("contact line loop did not close")
        loops.append(np.asarray(verts, dtype=np.int64))
    return loops


def loop_length(mesh: DropletMesh, loop: np.ndarray) -> float:
    """Length of a closed loop through mesh vertices, vox."""
    p = mesh.verts[loop]
    return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())
