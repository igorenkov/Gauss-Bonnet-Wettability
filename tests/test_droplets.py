import math

import numpy as np
import pytest

from gauss_bonnet.droplets import extract_droplet, find_droplets, label_droplets
from gauss_bonnet.io import (
    STATUS_BORDER,
    STATUS_NO_CONTACT,
    STATUS_OK,
    STATUS_TOO_SMALL,
    load_volume,
)
from gauss_bonnet.phases import OIL, SOLID, WATER
from gauss_bonnet.synthetic import spherical_cap


def _water(shape=(10, 10, 10)):
    return np.full(shape, WATER, dtype=np.uint8)


def test_find_droplets_single_cap_is_ok_with_exact_volume_and_centroid():
    cap = spherical_cap(12, math.radians(60))
    labels, droplets = find_droplets(cap.volume)
    assert len(droplets) == 1
    d = droplets[0]
    assert d.id == 1 and d.status == STATUS_OK
    assert d.touches_solid and not d.touches_border
    assert d.volume_vox == np.count_nonzero(cap.volume == OIL)
    assert np.count_nonzero(labels == 1) == d.volume_vox
    # the y, x centre matches the ball centre
    assert d.centroid[1] == pytest.approx(cap.center[1], abs=0.05)
    assert d.centroid[2] == pytest.approx(cap.center[2], abs=0.05)
    # bbox is the tight bounding box
    assert np.all(labels[d.bbox] >= 0) and np.any(labels[d.bbox][0] == 1)


def test_find_droplets_two_caps_side_by_side():
    cap = spherical_cap(8, math.radians(90)).volume
    vol = np.concatenate([cap, cap], axis=2)
    _, droplets = find_droplets(vol)
    assert [d.status for d in droplets] == [STATUS_OK, STATUS_OK]
    assert droplets[0].volume_vox == droplets[1].volume_vox
    assert droplets[0].centroid[2] < droplets[1].centroid[2]


def test_find_droplets_cap_cut_by_image_border_is_border():
    cap = spherical_cap(10, math.radians(90)).volume
    cut = cap[:, :, : cap.shape[2] // 2]  # droplet cut in half
    _, droplets = find_droplets(cut)
    assert len(droplets) == 1
    assert droplets[0].touches_border and droplets[0].status == STATUS_BORDER


def test_find_droplets_small_droplet_is_too_small():
    vol = _water()
    vol[:2] = SOLID
    vol[2, 5, 5] = OIL
    _, droplets = find_droplets(vol, min_voxels=27)
    assert droplets[0].status == STATUS_TOO_SMALL
    _, droplets = find_droplets(vol, min_voxels=1)
    assert droplets[0].status == STATUS_OK


def test_find_droplets_floating_droplet_has_no_contact():
    vol = _water((12, 12, 12))
    vol[:2] = SOLID
    vol[5:9, 4:8, 4:8] = OIL  # a water layer between oil and solid
    _, droplets = find_droplets(vol, min_voxels=1)
    assert not droplets[0].touches_solid
    assert droplets[0].status == STATUS_NO_CONTACT


def test_find_droplets_diagonal_solid_contact_does_not_count():
    vol = _water()
    vol[3, 3, 3] = SOLID
    vol[4, 4, 4] = OIL  # touching at a vertex only
    _, droplets = find_droplets(vol, min_voxels=1)
    assert not droplets[0].touches_solid


@pytest.mark.parametrize(("connectivity", "expected"), [(26, 1), (6, 2)])
def test_label_droplets_connectivity(connectivity, expected):
    vol = _water()
    vol[4, 4, 4] = OIL
    vol[5, 5, 4] = OIL  # edge neighbours
    _, n = label_droplets(vol, connectivity)
    assert n == expected


def test_find_droplets_without_oil_returns_empty():
    vol = _water()
    vol[:3] = SOLID
    labels, droplets = find_droplets(vol)
    assert droplets == [] and not labels.any()


@pytest.mark.parametrize("kwargs", [{"connectivity": 18}, {"min_voxels": 0}])
def test_find_droplets_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        find_droplets(_water(), **kwargs)


def test_extract_droplet_keeps_only_own_oil_and_maps_offset():
    cap = spherical_cap(8, math.radians(90)).volume
    vol = np.concatenate([cap, cap], axis=2)
    labels, droplets = find_droplets(vol)
    d = droplets[0]
    crop = extract_droplet(vol, labels, d, pad=2)
    assert crop.volume.dtype == np.uint8
    assert crop.volume.shape == tuple(n + 4 for n in d.bbox_shape)
    assert np.count_nonzero(crop.volume == OIL) == d.volume_vox
    # every oil voxel of the crop is a voxel of this droplet in the source volume
    for z, y, x in np.argwhere(crop.volume == OIL)[:50]:
        gz, gy, gx = np.array(crop.offset) + (z, y, x)
        assert labels[gz, gy, gx] == d.id
    # solid is preserved
    assert np.any(crop.volume == SOLID)


def test_extract_droplet_replaces_neighbouring_oil_with_water():
    vol = _water((8, 8, 14))
    vol[:2] = SOLID
    vol[2:5, 2:5, 2:5] = OIL
    vol[2:5, 2:5, 6:9] = OIL  # the second droplet falls into the padding of the first
    labels, droplets = find_droplets(vol, min_voxels=1)
    crop = extract_droplet(vol, labels, droplets[0], pad=3)
    assert np.count_nonzero(crop.volume == OIL) == droplets[0].volume_vox


def test_extract_droplet_pads_outside_image_with_water():
    cap = spherical_cap(10, math.radians(90)).volume
    cut = cap[:, :, : cap.shape[2] // 2]
    labels, droplets = find_droplets(cut)
    crop = extract_droplet(cut, labels, droplets[0], pad=2)
    assert np.all(crop.volume[:, :, -2:] == WATER)
    assert crop.offset[2] == droplets[0].bbox[2].start - 2


@pytest.mark.slow
def test_find_droplets_on_robuglas_45(data_dir):
    path = data_dir / "robuglas_45.npy"
    if not path.exists():
        pytest.skip(f"missing file {path}")
    vol = load_volume(path)
    _, droplets = find_droplets(vol)
    assert len(droplets) > 0
    assert sum(d.volume_vox for d in droplets) == np.count_nonzero(vol == OIL)
    assert any(d.status == STATUS_OK for d in droplets)
