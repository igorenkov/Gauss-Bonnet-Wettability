import math

import numpy as np
import pytest
from scipy import ndimage

from gauss_bonnet.phases import OIL, SOLID, WATER
from gauss_bonnet.synthetic import expected_theta_macro, spherical_cap

THETAS_DEG = [30, 45, 60, 90, 120, 150]


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_spherical_cap_labels_are_valid_phases(theta_deg):
    cap = spherical_cap(15, math.radians(theta_deg))
    assert cap.volume.dtype == np.uint8
    assert set(np.unique(cap.volume).tolist()) == {SOLID, WATER, OIL}


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_spherical_cap_oil_is_one_droplet_touching_solid_not_border(theta_deg):
    cap = spherical_cap(15, math.radians(theta_deg))
    oil = cap.volume == OIL
    _, n = ndimage.label(oil, structure=np.ones((3, 3, 3)))
    assert n == 1
    # oil touches the solid (a contact line exists)
    assert np.any(oil[1:] & (cap.volume[:-1] == SOLID))
    # oil does not touch the volume boundary
    for axis in range(3):
        assert not np.take(oil, 0, axis=axis).any()
        assert not np.take(oil, -1, axis=axis).any()


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_spherical_cap_oil_voxel_count_matches_analytic_volume(theta_deg):
    cap = spherical_cap(25, math.radians(theta_deg))
    n_oil = int(np.count_nonzero(cap.volume == OIL))
    assert n_oil == pytest.approx(cap.oil_volume, rel=0.03)


def test_spherical_cap_solid_is_flat_slab_below_plane():
    cap = spherical_cap(10, math.radians(60), solid_thickness=4)
    assert np.all(cap.volume[:4] == SOLID)
    assert not np.any(cap.volume[4:] == SOLID)


@pytest.mark.parametrize("theta_deg", THETAS_DEG)
def test_analytic_gauss_bonnet_identity_holds(theta_deg):
    cap = spherical_cap(20, math.radians(theta_deg))
    total = cap.sum_kappa_g_area_lv + cap.sum_kappa_g_area_ls + cap.deficit_curvature
    assert total == pytest.approx(4 * math.pi * cap.euler_surface / 2)
    assert cap.area_lv / cap.radius**2 == pytest.approx(cap.sum_kappa_g_area_lv)
    assert cap.area_ls == pytest.approx(math.pi * cap.height * (2 * cap.radius - cap.height))


@pytest.mark.parametrize(
    ("theta_deg", "macro_deg"),
    [(45, 26.360), (90, 90.0), (30, 12.058), (150, 167.942)],
)
def test_expected_theta_macro_known_values(theta_deg, macro_deg):
    got = math.degrees(expected_theta_macro(math.radians(theta_deg)))
    assert got == pytest.approx(macro_deg, abs=1e-3)
    cap = spherical_cap(20, math.radians(theta_deg))
    assert cap.theta_macro == pytest.approx(expected_theta_macro(cap.theta))


def test_spherical_cap_90deg_is_hemisphere_centred_on_plane():
    cap = spherical_cap(12, math.pi / 2)
    assert cap.center[0] == pytest.approx(cap.z_plane)
    assert cap.height == pytest.approx(cap.radius)
    assert cap.base_radius == pytest.approx(cap.radius)


def test_spherical_cap_offset_keeps_volume():
    a = spherical_cap(20, math.radians(45))
    b = spherical_cap(20, math.radians(45), offset=(0.3, 0.7, 0.1))
    na = np.count_nonzero(a.volume == OIL)
    nb = np.count_nonzero(b.volume == OIL)
    assert nb == pytest.approx(na, rel=0.02)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0, "theta": 1.0},
        {"radius": 10, "theta": 0.0},
        {"radius": 10, "theta": math.pi},
        {"radius": 10, "theta": 1.0, "solid_thickness": 0},
        {"radius": 10, "theta": 1.0, "margin": 0},
        {"radius": 10, "theta": 1.0, "offset": (0.0, 1.0, 0.0)},
        {"radius": 10, "theta": math.radians(179)},  # h < 1 voxel
    ],
)
def test_spherical_cap_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        spherical_cap(**kwargs)
