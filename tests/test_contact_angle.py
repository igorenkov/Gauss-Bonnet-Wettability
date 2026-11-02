import math
from dataclasses import replace

import numpy as np
import pytest

from gauss_bonnet.contact_angle import contact_angle, droplet_row, face_kappa_g_area
from gauss_bonnet.curvature import face_curvature, vertex_angle_defects
from gauss_bonnet.droplets import extract_droplet, find_droplets
from gauss_bonnet.io import (
    DROPLET_COLUMNS,
    STATUS_BORDER,
    STATUS_NO_CONTACT,
    STATUS_OK,
    STATUS_OUT_OF_RANGE,
    load_volume,
    read_droplets_csv,
    write_droplets_csv,
)
from gauss_bonnet.mesh import ROLE_LS, ROLE_LV, mesh_droplet
from gauss_bonnet.phases import OIL, SOLID, WATER
from gauss_bonnet.synthetic import spherical_cap
from gauss_bonnet.topology import mesh_topology


def _pipeline(vol, **kw):
    labels, droplets = find_droplets(vol, min_voxels=1)
    d = droplets[0]
    m = mesh_droplet(extract_droplet(vol, labels, d))
    t = mesh_topology(m)
    c = face_curvature(m)
    return d, m, t, c, contact_angle(m, t, c, **kw)


def _cap(theta_deg, radius=20):
    return _pipeline(spherical_cap(radius, math.radians(theta_deg)).volume)


def test_hemisphere_gives_right_angle():
    *_, a = _cap(90)
    assert a.status == STATUS_OK and a.n_loops == 1 and a.euler_droplet == 1
    assert a.theta_macro_deg == pytest.approx(90, abs=5)
    assert a.sum_kappa_g_area_ls == pytest.approx(0, abs=1e-9)


def test_theta_macro_grows_with_contact_angle():
    thetas = [_cap(th)[-1].theta_macro_deg for th in (30, 60, 90, 120, 150)]
    assert np.all(np.diff(thetas) > 0)


def test_raw_deficit_equals_defect_on_contact_line():
    # without sign correction discrete Gauss-Bonnet is exact: Θ = sum of contact line vertex defects
    _, m, _, _, a = _cap(60, radius=12)
    roles = m.vertex_roles
    line = ~np.isin(roles, (ROLE_LV, ROLE_LS))
    assert a.deficit_curvature_raw == pytest.approx(vertex_angle_defects(m)[line].sum(), abs=1e-9)
    assert a.deficit_curvature == pytest.approx(a.deficit_curvature_raw)


def _fake_curvature(c, kappa_g, kappa_m):
    return replace(c, kappa_g=np.asarray(kappa_g, float), kappa_m=np.asarray(kappa_m, float))


def test_sign_correction_flips_concave_faces_with_positive_gaussian_curvature():
    _, m, _, c, _ = _cap(90, radius=8)
    n = len(m.faces)
    kg = np.full(n, 0.01)
    km = np.where(np.arange(n) % 2 == 0, -0.1, 0.1)
    fake = _fake_curvature(c, kg, km)
    values, flipped = face_kappa_g_area(m, fake, sign_correction=True)
    assert flipped.sum() == (n + 1) // 2
    np.testing.assert_allclose(values, np.where(km < 0, -1, 1) * kg * m.face_areas)
    # κ_G < 0 is not corrected; flat ones (|κ_G| ≤ flat_eps) give 0
    values, flipped = face_kappa_g_area(m, _fake_curvature(c, -kg, km), sign_correction=True)
    assert not flipped.any()
    values, flipped = face_kappa_g_area(m, fake, sign_correction=True, flat_eps=0.02)
    assert not flipped.any() and not values.any()


def test_sign_correction_is_off_by_default():
    _, m, t, c, a = _cap(60, radius=12)
    assert a.n_flipped == 0
    assert a.deficit_curvature == pytest.approx(a.deficit_curvature_raw)
    on = contact_angle(m, t, c, sign_correction=True)
    assert on.n_flipped > 0


def test_floating_droplet_has_no_contact():
    vol = np.full((16, 16, 16), WATER, dtype=np.uint8)
    vol[:2] = SOLID
    z, y, x = np.indices(vol.shape)
    vol[(z - 9) ** 2 + (y - 8) ** 2 + (x - 8) ** 2 <= 16] = OIL
    *_, a = _pipeline(vol)
    assert a.status == STATUS_NO_CONTACT and a.theta_macro is None and a.theta_macro_deg is None
    # Σ κ_G A of a closed sphere = 4π, hence Θ ≈ 0
    assert a.deficit_curvature_raw == pytest.approx(0, abs=1e-9)


def test_out_of_range_keeps_value():
    _, m, t, c, _ = _cap(90, radius=8)
    big = _fake_curvature(c, np.full(len(m.faces), 1.0), np.full(len(m.faces), 1.0))
    a = contact_angle(m, t, big)
    assert a.status == STATUS_OUT_OF_RANGE and a.theta_macro < 0


def test_euler_mode_ball_ignores_mesh_topology():
    vol = np.full((10, 24, 24), WATER, dtype=np.uint8)
    vol[:3] = SOLID
    z, y, x = np.indices(vol.shape)
    rho = np.hypot(y - 11.5, x - 11.5)
    vol[(z >= 3) & (z < 6) & (rho >= 4) & (rho <= 9)] = OIL  # torus on the solid: χ(D) = 0, N = 2
    _, m, t, c, a = _pipeline(vol)
    b = contact_angle(m, t, c, euler="ball")
    assert a.euler_droplet == 0 and b.euler_droplet == 1 and a.n_loops == 2
    assert b.theta_macro - a.theta_macro == pytest.approx(4 * math.pi / (4 * 2))


@pytest.mark.parametrize("kwargs", [{"euler": "torus"}, {"flat_eps": -1.0}])
def test_contact_angle_rejects_bad_parameters(kwargs):
    _, m, t, c, _ = _cap(90, radius=6)
    with pytest.raises(ValueError):
        contact_angle(m, t, c, **kwargs)


def test_droplet_row_round_trips_through_csv(tmp_path):
    d, m, t, _, a = _cap(60, radius=10)
    row = droplet_row(d, m, t, a, voxel_size=4.2)
    assert set(row) <= set(DROPLET_COLUMNS)
    assert row["area_lv_um2"] == pytest.approx(m.area_lv * 4.2**2)
    assert row["theta_macro_deg"] == pytest.approx(a.theta_macro_deg)
    assert row["status"] == STATUS_OK
    skipped = replace(d, id=2, status=STATUS_BORDER, touches_border=True)
    path = write_droplets_csv([row, droplet_row(skipped)], tmp_path / "droplets.csv")
    got = read_droplets_csv(path)
    assert got[0]["n_loops"] == 1 and got[0]["theta_macro_deg"] == pytest.approx(row["theta_macro_deg"])
    assert got[1]["status"] == STATUS_BORDER and got[1]["theta_macro_deg"] is None
    assert got[1]["area_lv"] is None


def test_droplet_row_rejects_bad_voxel_size():
    d, *_ = _cap(90, radius=6)
    with pytest.raises(ValueError):
        droplet_row(d, voxel_size=0)


@pytest.mark.slow
def test_contact_angle_on_robuglas_45(data_dir):
    path = data_dir / "robuglas_45.npy"
    if not path.exists():
        pytest.skip(f"missing file {path}")
    vol = load_volume(path)
    labels, droplets = find_droplets(vol)
    statuses = []
    for d in droplets:
        if d.status != STATUS_OK:
            continue
        m = mesh_droplet(extract_droplet(vol, labels, d))
        t = mesh_topology(m)
        a = contact_angle(m, t, face_curvature(m))
        assert np.isfinite(a.deficit_curvature), d.id
        statuses.append(a.status)
    assert statuses.count(STATUS_OK) >= 0.9 * len(statuses)


def test_droplet_row_for_water_droplet_reports_angle_through_water():
    d, m, t, _, a = _cap(60, radius=10)
    oil = droplet_row(d, m, t, a)
    water = droplet_row(d, m, t, a, phase="water")
    assert oil["phase"] == "oil" and water["phase"] == "water"
    assert water["theta_macro_deg"] == pytest.approx(180 - oil["theta_macro_deg"])
    assert water["theta_macro_raw_deg"] == pytest.approx(180 - oil["theta_macro_raw_deg"])
    assert water["deficit_curvature"] == oil["deficit_curvature"]  # Θ is a property of the droplet itself
    assert droplet_row(replace(d, status=STATUS_BORDER), phase="water")["phase"] == "water"
    with pytest.raises(ValueError, match="phase"):
        droplet_row(d, phase="gas")
