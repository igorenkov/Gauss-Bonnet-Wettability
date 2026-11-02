import math

import numpy as np
import pytest

from gauss_bonnet.droplets import extract_droplet, find_droplets
from gauss_bonnet.io import STATUS_OK, load_volume
from gauss_bonnet.mesh import (
    ROLE_CONTACT,
    ROLE_FIXED,
    ROLE_LS,
    build_mesh,
    mesh_droplet,
    smooth_mesh,
)
from gauss_bonnet.phases import OIL, SOLID, WATER
from gauss_bonnet.synthetic import spherical_cap

THETAS_DEG = [30, 45, 90, 135]


def _cap_mesh(radius, theta_deg, **kw):
    cap = spherical_cap(radius, math.radians(theta_deg), **kw)
    labels, droplets = find_droplets(cap.volume)
    return cap, build_mesh(extract_droplet(cap.volume, labels, droplets[0]))


def _cap_mesh_raw(radius, theta_deg):
    cap = spherical_cap(radius, math.radians(theta_deg))
    labels, droplets = find_droplets(cap.volume)
    crop = extract_droplet(cap.volume, labels, droplets[0])
    return cap, build_mesh(crop, triple_line=False)


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_build_mesh_is_closed_oriented_outward_with_voxel_volume(theta_deg):
    cap, m = _cap_mesh(20, theta_deg)
    assert m.is_closed_manifold()
    n_oil = np.count_nonzero(cap.volume == OIL)
    assert m.enclosed_volume == pytest.approx(n_oil, rel=0.02)  # > 0 ⇒ outward normals
    # I_lv normals point away from the ball centre on average
    centers = m.verts[m.faces[m.is_lv]].mean(axis=1)
    radial = centers - np.array(cap.center)
    assert np.mean(np.einsum("ij,ij->i", radial, m.face_normals[m.is_lv]) > 0) > 0.95


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_build_mesh_labels_ls_faces_on_solid_plane(theta_deg):
    cap, m = _cap_mesh(20, theta_deg)
    ls_verts = np.unique(m.faces[m.is_ls])
    # all I_ls vertices lie on the substrate plane and I_ls normals point down (into the solid)
    np.testing.assert_allclose(m.verts[ls_verts, 0], cap.z_plane, atol=1e-9)
    assert np.all(m.face_normals[m.is_ls, 0] < -0.99)
    assert m.is_lv.sum() + m.is_ls.sum() == len(m.faces)


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_contact_line_is_single_closed_loop_on_plane(theta_deg):
    cap, m = _cap_mesh(20, theta_deg)
    line = m.contact_edges
    verts_on_line, deg = np.unique(line.ravel(), return_counts=True)
    assert len(line) > 0 and np.all(deg == 2)
    np.testing.assert_allclose(m.verts[verts_on_line, 0], cap.z_plane, atol=1e-9)
    roles = m.vertex_roles
    assert np.all(roles[verts_on_line] == ROLE_CONTACT)
    assert not np.any(roles == ROLE_FIXED)


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_taubin_smoothing_matches_analytic_areas_and_volume(theta_deg):
    cap, m = _cap_mesh(20, theta_deg)
    s = smooth_mesh(m)  # taubin by default
    assert s.enclosed_volume == pytest.approx(cap.oil_volume, rel=0.02)
    assert s.area_lv == pytest.approx(cap.area_lv, rel=0.03)
    # the contact spot is an inscribed polygon: the error is at most a 1 vox wide band along the edge
    assert abs(s.area_ls - cap.area_ls) <= cap.contact_line_length * 1.0


def test_smoothing_keeps_solid_patch_and_contact_line_in_plane():
    cap, m = _cap_mesh(15, 60)
    s = smooth_mesh(m, iterations=30)
    on_plane = np.isin(m.vertex_roles, (ROLE_LS, ROLE_CONTACT))
    np.testing.assert_allclose(s.verts[on_plane, 0], cap.z_plane, atol=1e-9)
    # lv vertices moved, so smoothing does work
    assert np.abs(s.verts[~on_plane] - m.verts[~on_plane]).max() > 0.05


def test_smoothing_preserves_connectivity_and_labels():
    _, m = _cap_mesh(12, 45)
    s = smooth_mesh(m, method="laplacian", alpha=0.3, iterations=5)
    assert s.faces is m.faces and s.face_phase is m.face_phase
    assert s.is_closed_manifold()


def test_laplacian_shrinks_more_than_taubin():
    cap, m = _cap_mesh(10, 90)
    lap = smooth_mesh(m, method="laplacian", iterations=20)
    tau = smooth_mesh(m, method="taubin", iterations=20)
    assert lap.enclosed_volume < tau.enclosed_volume
    assert tau.enclosed_volume == pytest.approx(cap.oil_volume, rel=0.03)


def test_zero_iterations_is_identity():
    _, m = _cap_mesh(8, 90)
    np.testing.assert_array_equal(smooth_mesh(m, iterations=0).verts, m.verts)


def test_mesh_coordinates_are_global():
    cap = spherical_cap(8, math.radians(90))
    # shift the droplet by (0, 6, 11): pad the volume with edge values (solid below, water above)
    shifted = np.pad(cap.volume, ((0, 0), (6, 0), (11, 0)), mode="edge")
    labels, droplets = find_droplets(shifted)
    m = build_mesh(extract_droplet(shifted, labels, droplets[0]))
    _, m0 = _cap_mesh(8, 90)
    np.testing.assert_allclose(m.verts.mean(axis=0), m0.verts.mean(axis=0) + (0, 6, 11))


def _single_voxel_contact():
    vol = np.full((10, 10, 10), WATER, dtype=np.uint8)
    vol[:3] = SOLID
    vol[3:6, 3:6, 3:6] = OIL
    vol[3, 3:6, 3:6] = WATER  # the droplet is lifted above the solid...
    vol[3, 4, 4] = OIL  # ...and touches it with one voxel face
    labels, droplets = find_droplets(vol, min_voxels=1)
    assert droplets[0].touches_solid
    return extract_droplet(vol, labels, droplets[0])


def test_single_voxel_solid_contact_gives_small_ls_patch():
    m = build_mesh(_single_voxel_contact())
    # triple line on the voxel face edges: the spot is a rhombus with vertices at edge midpoints
    assert m.area_ls == pytest.approx(0.5)
    assert len(m.contact_edges) == 4
    assert m.is_closed_manifold()


@pytest.mark.parametrize("floor", [WATER, SOLID])
def test_ambiguous_cube_center_vertex_is_handled(floor):
    # cube configuration from robuglas_135 (droplet 64): at level 0.5 Lewiner puts a vertex
    # at the cube centre; build_mesh (level MC_LEVEL) must give a closed mesh
    vol = np.full((6, 6, 6), WATER, dtype=np.uint8)
    vol[0] = floor
    cube = np.array([[[1, 0], [1, 0]], [[1, 1], [0, 1]]], dtype=bool)
    vol[2:4, 2:4, 2:4][cube] = OIL
    if floor == SOLID:
        vol[1, 2:4, 2:4] = OIL  # contact with the solid
    labels, droplets = find_droplets(vol, min_voxels=1)
    crop = extract_droplet(vol, labels, droplets[0])
    from skimage import measure

    raw, *_ = measure.marching_cubes((crop.volume == OIL).astype(np.float32), 0.5,
                                     method="lewiner", allow_degenerate=False)
    assert np.any(np.all(np.abs(raw - np.floor(raw) - 0.5) < 1e-6, axis=1)), "no cube centre"
    m = build_mesh(crop)
    assert m.is_closed_manifold()
    assert set(np.unique(m.face_phase)) <= {WATER, SOLID}


# A 26-connected droplet of 11 voxels for which Lewiner at level 0.5 resolves saddle point
# ties inconsistently: mesh χ = 1 (the surface is not closed).
_TIE_BLOB = np.array([[[0, 1, 1], [1, 0, 0], [0, 0, 0]],
                      [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                      [[1, 1, 1], [1, 0, 0], [0, 1, 0]]], dtype=bool)


def _euler(verts, faces):
    edges = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    return len(verts) - len(edges) + len(faces)


def test_saddle_ties_do_not_create_false_tunnels():
    from skimage import measure

    vol = np.full((7, 7, 7), WATER, dtype=np.uint8)
    vol[2:5, 2:5, 2:5][_TIE_BLOB] = OIL
    labels, droplets = find_droplets(vol, min_voxels=1)
    assert len(droplets) == 1
    crop = extract_droplet(vol, labels, droplets[0])
    raw, raw_faces, *_ = measure.marching_cubes((crop.volume == OIL).astype(np.float32), 0.5,
                                                method="lewiner", allow_degenerate=False)
    assert _euler(raw, raw_faces) != 2, "the defect no longer reproduces at level 0.5"
    m = build_mesh(crop)
    assert m.is_closed_manifold() and _euler(m.verts, m.faces) == 2
    frac = m.verts - np.floor(m.verts)
    assert np.all(np.isclose(frac, 0) | np.isclose(frac, 0.5))  # vertices at edge midpoints


def test_without_triple_line_single_voxel_contact_gives_no_ls_faces():
    m = build_mesh(_single_voxel_contact(), triple_line=False)
    assert m.area_ls == 0 and len(m.contact_edges) == 0


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_triple_line_moves_contact_line_to_voxel_corners(theta_deg):
    cap, m = _cap_mesh(20, theta_deg)
    _, m0 = _cap_mesh_raw(20, theta_deg)
    # without the correction the contact line passes through the centres of the outermost oil
    # voxels; with it some vertices move to voxel edges, so the spot is ≈ half a voxel wider
    assert m.area_ls > m0.area_ls
    assert abs(m.area_ls - cap.area_ls) <= cap.contact_line_length * 1.0
    assert len(m.verts) == len(m0.verts) and np.array_equal(m.faces, m0.faces)


@pytest.mark.parametrize(
    "kwargs", [{"alpha": 0.0}, {"alpha": 1.5}, {"iterations": -1}, {"method": "gauss"}]
)
def test_smooth_mesh_rejects_bad_parameters(kwargs):
    _, m = _cap_mesh(6, 90)
    with pytest.raises(ValueError):
        smooth_mesh(m, **kwargs)


@pytest.mark.slow
def test_mesh_all_ok_droplets_of_robuglas_45(data_dir):
    path = data_dir / "robuglas_45.npy"
    if not path.exists():
        pytest.skip(f"missing file {path}")
    vol = load_volume(path)
    labels, droplets = find_droplets(vol)
    for d in droplets:
        if d.status != STATUS_OK:
            continue
        m = mesh_droplet(extract_droplet(vol, labels, d))
        assert m.is_closed_manifold(), d.id
        assert m.enclosed_volume > 0, d.id
