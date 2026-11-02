import math
from dataclasses import replace

import numpy as np
import pytest

from gauss_bonnet.curvature import (
    face_angle_defect_curvature,
    face_curvature,
    vertex_angle_defects,
)
from gauss_bonnet.droplets import extract_droplet, find_droplets
from gauss_bonnet.io import STATUS_OK, load_volume
from gauss_bonnet.mesh import ROLE_LS, build_mesh, mesh_droplet, smooth_mesh
from gauss_bonnet.phases import OIL, SOLID, WATER
from gauss_bonnet.synthetic import spherical_cap
from gauss_bonnet.topology import mesh_topology


def _mesh(vol, **kw):
    labels, droplets = find_droplets(vol, min_voxels=1)
    assert len(droplets) == 1
    return build_mesh(extract_droplet(vol, labels, droplets[0]), **kw)


def _ball(shape, center, radius):
    z, y, x = np.indices(shape)
    c = np.asarray(center, dtype=float).reshape(3, 1, 1, 1)
    return ((np.stack([z, y, x]) - c) ** 2).sum(axis=0) <= radius**2


def _sphere_mesh(radius):
    n = int(2 * radius + 8)
    vol = np.full((n, n, n), WATER, dtype=np.uint8)
    vol[_ball(vol.shape, (n / 2,) * 3, radius)] = OIL
    return smooth_mesh(_mesh(vol))


def _theta_deg(sum_kappa_g_area, n_loops=1):
    return math.degrees((4 * math.pi - sum_kappa_g_area) / (4 * n_loops))


def test_sphere_quadric_curvatures_match_radius():
    r = 20
    m = _sphere_mesh(r)
    c = face_curvature(m, method="quadric")
    assert c.fit_ok.all()
    assert c.sum_kappa_g_area(m) == pytest.approx(4 * math.pi, rel=0.03)
    assert np.median(c.kappa_m) == pytest.approx(1 / r, rel=0.05)  # convex ⇒ κ_M > 0
    # pointwise κ_G is noisy (median ≈ 13 % low), the area sum converges better
    assert np.median(c.kappa_g) == pytest.approx(1 / r**2, rel=0.15)
    assert np.all(c.kappa1 >= c.kappa2)
    np.testing.assert_allclose(c.kappa1 * c.kappa2, c.kappa_g, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose((c.kappa1 + c.kappa2) / 2, c.kappa_m, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("radius", [4, 6, 10, 20])
def test_angle_defect_on_closed_surface_is_exact(radius):
    m = _sphere_mesh(radius)
    c = face_curvature(m, method="angle_defect")
    # discrete Gauss-Bonnet: Σ defects = 2π χ(∂D) = 4π on a sphere, at any size
    assert c.sum_kappa_g_area(m) == pytest.approx(4 * math.pi, abs=1e-9)


def test_quadric_overestimates_on_small_spheres_with_wide_neighbourhood():
    # known limitation of the quadric fit (see the module docstring)
    m = _sphere_mesh(4)
    narrow = face_curvature(m, 2, method="quadric").sum_kappa_g_area(m)
    wide = face_curvature(m, 7, method="quadric").sum_kappa_g_area(m)
    assert 4 * math.pi < narrow < wide


def test_vertex_angle_defects_sum_to_euler_characteristic_on_torus():
    vol = np.full((12, 20, 20), WATER, dtype=np.uint8)
    z, y, x = np.indices(vol.shape)
    vol[np.hypot(np.hypot(y - 9.5, x - 9.5) - 5.5, z - 6) <= 2.5] = OIL
    m = smooth_mesh(_mesh(vol))
    assert mesh_topology(m).euler_surface == 0
    assert vertex_angle_defects(m).sum() == pytest.approx(0, abs=1e-9)
    kg = face_curvature(m).kappa_g
    # the outer side of a torus is convex (κ_G > 0), the inner side a saddle (κ_G < 0)
    centres = m.verts[m.faces].mean(axis=1)
    rho = np.hypot(centres[:, 1] - 9.5, centres[:, 2] - 9.5)
    assert np.median(kg[rho > 7]) > 0 > np.median(kg[rho < 4])


def test_concave_cavity_has_negative_mean_curvature():
    vol = np.full((20, 20, 20), WATER, dtype=np.uint8)
    vol[_ball(vol.shape, (10, 10, 10), 8)] = OIL
    vol[_ball(vol.shape, (10, 10, 10), 4)] = WATER  # cavity: the normal out of the oil points into the cavity
    m = smooth_mesh(_mesh(vol))
    c = face_curvature(m)
    dist = np.linalg.norm(m.verts[m.faces].mean(axis=1) - 10, axis=1)
    inner = dist < 6
    assert np.median(c.kappa_m[inner]) < 0 < np.median(c.kappa_m[~inner])
    assert np.median(c.kappa_g[inner]) > 0  # sphere from inside: κ_G > 0, κ_M < 0 (sign correction, Alg. 1)


def test_cylinder_has_zero_gaussian_and_half_mean_curvature():
    r = 8
    vol = np.full((30, 24, 24), WATER, dtype=np.uint8)
    vol[:3] = SOLID
    vol[-3:] = SOLID
    z, y, x = np.indices(vol.shape)
    vol[(z >= 3) & (z < 27) & (np.hypot(y - 11.5, x - 11.5) <= r)] = OIL  # cylindrical bridge
    m = smooth_mesh(_mesh(vol))
    c = face_curvature(m)
    zc = m.verts[m.faces].mean(axis=1)[:, 0]
    mid = m.is_lv & (zc > 10) & (zc < 20)
    assert np.median(c.kappa_m[mid]) == pytest.approx(1 / (2 * r), rel=0.1)
    assert abs(np.median(c.kappa_g[mid])) < 0.1 / r**2


@pytest.mark.parametrize("theta_deg", [45, 90, 135])
def test_solid_patch_on_plane_is_flat(theta_deg):
    cap = spherical_cap(15, math.radians(theta_deg))
    m = smooth_mesh(_mesh(cap.volume))
    for method in ("quadric", "angle_defect"):
        c = face_curvature(m, method=method)
        assert np.abs(c.kappa_g[m.is_ls]).max() < 1e-9
        assert c.sum_kappa_g_area(m, SOLID) == pytest.approx(0, abs=1e-9)


def test_lv_curvature_does_not_see_solid_patch():
    # key rule: the fit on I_lv does not use interior I_ls vertices
    cap = spherical_cap(12, math.radians(90))
    m = smooth_mesh(_mesh(cap.volume))
    rng = np.random.default_rng(0)
    verts = m.verts.copy()
    ls = m.vertex_roles == ROLE_LS
    verts[ls] += rng.normal(scale=0.3, size=(ls.sum(), 3))

    moved = replace(m, verts=verts)
    for method in ("quadric", "angle_defect"):
        a, b = face_curvature(m, method=method), face_curvature(moved, method=method)
        np.testing.assert_allclose(a.kappa_g[m.is_lv], b.kappa_g[m.is_lv], atol=1e-12)
        np.testing.assert_allclose(a.kappa_m[m.is_lv], b.kappa_m[m.is_lv], atol=1e-12)


@pytest.mark.parametrize("radius", [10, 20])
def test_hemisphere_gives_right_angle_only_with_triple_line_correction(radius):
    cap = spherical_cap(radius, math.radians(90))
    fixed = smooth_mesh(_mesh(cap.volume))
    raw = smooth_mesh(_mesh(cap.volume, triple_line=False))
    for m, ok in ((fixed, True), (raw, False)):
        s = face_curvature(m, method="angle_defect").sum_kappa_g_area(m)
        assert (abs(_theta_deg(s) - 90) < 6) == ok


def test_angle_defect_face_values_sum_to_interior_defects():
    cap = spherical_cap(10, math.radians(60))
    m = smooth_mesh(_mesh(cap.volume))
    kg = face_angle_defect_curvature(m)
    roles = m.vertex_roles
    interior = (roles == 0) | (roles == ROLE_LS)
    assert np.sum(kg * m.face_areas) == pytest.approx(vertex_angle_defects(m)[interior].sum())


def test_default_method_is_angle_defect():
    m = _sphere_mesh(6)
    np.testing.assert_array_equal(
        face_curvature(m).kappa_g, face_curvature(m, method="angle_defect").kappa_g
    )


@pytest.mark.parametrize("kwargs", [{"n_layer": 0}, {"method": "spline"}])
def test_face_curvature_rejects_bad_parameters(kwargs):
    m = _sphere_mesh(5)
    with pytest.raises(ValueError):
        face_curvature(m, **kwargs)


@pytest.mark.slow
def test_curvature_of_robuglas_45(data_dir):
    path = data_dir / "robuglas_45.npy"
    if not path.exists():
        pytest.skip(f"missing file {path}")
    vol = load_volume(path)
    labels, droplets = find_droplets(vol)
    for d in droplets:
        if d.status != STATUS_OK:
            continue
        m = mesh_droplet(extract_droplet(vol, labels, d))
        c = face_curvature(m)
        assert np.all(np.isfinite(c.kappa_g)) and np.all(np.isfinite(c.kappa_m)), d.id
        t = mesh_topology(m)
        total = vertex_angle_defects(m).sum()
        assert total == pytest.approx(2 * math.pi * t.euler_surface, abs=1e-6), d.id
