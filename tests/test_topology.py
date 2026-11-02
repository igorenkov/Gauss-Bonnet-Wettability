import math

import numpy as np
import pytest

from gauss_bonnet.droplets import extract_droplet, find_droplets
from gauss_bonnet.io import STATUS_OK, load_volume
from gauss_bonnet.mesh import DropletMesh, build_mesh, mesh_droplet, smooth_mesh
from gauss_bonnet.phases import OIL, SOLID, WATER
from gauss_bonnet.synthetic import spherical_cap
from gauss_bonnet.topology import contact_loops, loop_length, mesh_topology


def _mesh(vol):
    labels, droplets = find_droplets(vol, min_voxels=1)
    assert len(droplets) == 1
    return build_mesh(extract_droplet(vol, labels, droplets[0]))


def _ball(shape, center, radius):
    z, y, x = np.indices(shape)
    c = np.asarray(center, dtype=float).reshape(3, 1, 1, 1)
    return ((np.stack([z, y, x]) - c) ** 2).sum(axis=0) <= radius**2


def _assert_loops_cover_contact_line(m, topo):
    assert sum(len(lp) for lp in topo.loops) == len(m.contact_edges)
    line = {tuple(e) for e in m.contact_edges}
    for lp in topo.loops:
        for a, b in zip(lp, np.roll(lp, -1)):
            assert (min(a, b), max(a, b)) in line


@pytest.mark.parametrize("theta_deg", [30, 45, 90, 135])
def test_spherical_cap_is_ball_with_one_loop(theta_deg):
    cap = spherical_cap(20, math.radians(theta_deg))
    m = _mesh(cap.volume)
    t = mesh_topology(m)
    assert t.euler_surface == cap.euler_surface == 2
    assert t.euler_droplet == 1 and t.is_ball and t.n_components == 1
    assert t.n_loops == cap.n_loops == 1 and t.n_junctions == 0
    _assert_loops_cover_contact_line(m, t)
    # after smoothing the loop length is close to the contact circle length
    s = smooth_mesh(m)
    assert loop_length(s, t.loops[0]) == pytest.approx(cap.contact_line_length, rel=0.1)


def test_loop_follows_orientation_of_solid_patch():
    cap = spherical_cap(15, math.radians(60))
    m = _mesh(cap.volume)
    (loop,) = contact_loops(m)
    # the spot normal points down (−z), so its boundary runs clockwise in the (y, x) plane
    # seen from above: the sign of the polygon area is fixed
    p = m.verts[loop][:, 1:]
    signed = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    (loop_lv,) = contact_loops(m, WATER)
    q = m.verts[loop_lv][:, 1:]
    signed_lv = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    assert signed * signed_lv < 0  # the shared boundary of the two regions runs in opposite directions
    assert abs(signed) == pytest.approx(m.area_ls, rel=1e-9)


def test_smoothing_does_not_change_topology():
    cap = spherical_cap(12, math.radians(120))
    m = _mesh(cap.volume)
    t0, t1 = mesh_topology(m), mesh_topology(smooth_mesh(m, iterations=30))
    assert (t0.euler_surface, t0.n_loops) == (t1.euler_surface, t1.n_loops) == (2, 1)


def test_floating_sphere_has_no_loops():
    vol = np.full((16, 16, 16), WATER, dtype=np.uint8)
    vol[:2] = SOLID
    vol[_ball(vol.shape, (9, 8, 8), 4)] = OIL
    t = mesh_topology(_mesh(vol))
    assert t.is_ball and t.n_loops == 0 and t.loops == ()


def test_single_voxel_contact_has_one_loop():
    vol = np.full((10, 10, 10), WATER, dtype=np.uint8)
    vol[:3] = SOLID
    vol[3:6, 3:6, 3:6] = OIL
    vol[3, 3:6, 3:6] = WATER
    vol[3, 4, 4] = OIL  # touches the solid with one voxel face
    assert mesh_topology(_mesh(vol)).n_loops == 1


def test_torus_has_zero_euler_characteristic():
    vol = np.full((12, 20, 20), WATER, dtype=np.uint8)
    z, y, x = np.indices(vol.shape)
    rho = np.hypot(y - 9.5, x - 9.5)
    vol[np.hypot(rho - 5.5, z - 6) <= 2.5] = OIL  # torus in water, no solid
    t = mesh_topology(_mesh(vol))
    assert t.euler_surface == 0 and t.euler_droplet == 0 and not t.is_ball
    assert t.n_loops == 0


def test_ring_on_plane_has_annulus_contact_with_two_loops():
    vol = np.full((10, 24, 24), WATER, dtype=np.uint8)
    vol[:3] = SOLID
    z, y, x = np.indices(vol.shape)
    rho = np.hypot(y - 11.5, x - 11.5)
    vol[(z >= 3) & (z < 6) & (rho >= 4) & (rho <= 9)] = OIL  # washer-shaped ring on the solid
    m = _mesh(vol)
    t = mesh_topology(m)
    assert t.euler_surface == 0  # torus
    assert t.n_loops == 2  # outer and inner boundaries of the ring spot
    _assert_loops_cover_contact_line(m, t)
    outer, inner = sorted((loop_length(m, lp) for lp in t.loops), reverse=True)
    assert outer > inner > 0


def test_bridge_between_two_plates_has_two_loops():
    vol = np.full((14, 14, 14), WATER, dtype=np.uint8)
    vol[:3] = SOLID
    vol[-3:] = SOLID
    z, y, x = np.indices(vol.shape)
    vol[(z >= 3) & (z < 11) & (np.hypot(y - 6.5, x - 6.5) <= 3.5)] = OIL  # cylindrical bridge
    m = _mesh(vol)
    t = mesh_topology(m)
    assert t.is_ball and t.n_loops == 2
    _assert_loops_cover_contact_line(m, t)
    # the loops lie on different plates
    heights = sorted(float(m.verts[lp, 0].mean()) for lp in t.loops)
    assert heights == pytest.approx([2.5, 10.5])


def test_droplet_with_cavity_has_two_components():
    vol = np.full((16, 16, 16), WATER, dtype=np.uint8)
    vol[_ball(vol.shape, (8, 8, 8), 6)] = OIL
    vol[_ball(vol.shape, (8, 8, 8), 2.5)] = WATER  # water bubble inside the droplet
    t = mesh_topology(_mesh(vol))
    assert t.n_components == 2 and t.euler_surface == 4 and not t.is_ball


def _bowtie_mesh():
    """Ball mesh where two non-adjacent faces of one vertex are assigned to I_ls."""
    vol = np.full((14, 14, 14), WATER, dtype=np.uint8)
    vol[_ball(vol.shape, (7, 7, 7), 4)] = OIL
    m = _mesh(vol)
    valence = np.bincount(m.faces.ravel())
    v = int(np.argmax(valence))
    assert valence[v] >= 4
    fan = np.flatnonzero(np.any(m.faces == v, axis=1))
    # order the face fan around v and take two opposite faces
    nxt = {}
    for f in fan:
        k = int(np.flatnonzero(m.faces[f] == v)[0])
        nxt[int(m.faces[f, (k + 1) % 3])] = (int(m.faces[f, (k + 2) % 3]), int(f))
    order, a = [], next(iter(nxt))
    for _ in range(len(fan)):
        a, f = nxt[a]
        order.append(f)
    phase = np.full(len(m.faces), WATER, dtype=np.uint8)
    phase[[order[0], order[len(order) // 2]]] = SOLID
    return DropletMesh(verts=m.verts, faces=m.faces, face_phase=phase), v


def test_junction_vertex_splits_loops_by_solid_patches():
    m, v = _bowtie_mesh()
    t = mesh_topology(m)
    assert t.n_junctions == 1
    assert t.n_loops == 2  # two I_ls spots → two loops
    assert all(len(lp) == 3 and v in lp for lp in t.loops)
    _assert_loops_cover_contact_line(m, t)
    # walking over I_lv joins them into one figure-eight through vertex v
    (eight,) = contact_loops(m, WATER)
    assert len(eight) == 6 and np.count_nonzero(eight == v) == 2


def test_contact_loops_rejects_bad_region():
    m = _mesh(spherical_cap(6, math.radians(90)).volume)
    with pytest.raises(ValueError):
        contact_loops(m, OIL)


@pytest.mark.slow
def test_topology_of_robuglas_45(data_dir):
    path = data_dir / "robuglas_45.npy"
    if not path.exists():
        pytest.skip(f"missing file {path}")
    vol = load_volume(path)
    labels, droplets = find_droplets(vol)
    n_ball = 0
    for d in droplets:
        if d.status != STATUS_OK:
            continue
        m = mesh_droplet(extract_droplet(vol, labels, d))
        t = mesh_topology(m)
        _assert_loops_cover_contact_line(m, t)
        assert (t.n_loops == 0) == (m.area_ls == 0), d.id
        n_ball += t.is_ball
    assert n_ball > 0
