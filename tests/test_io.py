import json
import math

import numpy as np
import pytest

from gauss_bonnet.io import (
    DROPLET_COLUMNS,
    has_oil,
    load_volume,
    phase_fractions,
    read_droplets_csv,
    read_summary_json,
    summarize_droplets,
    validate_volume,
    write_droplets_csv,
    write_summary_json,
)
from gauss_bonnet.phases import OIL, SOLID, WATER
from gauss_bonnet.synthetic import spherical_cap

# ------------------------------------------------------------------ input


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int64, np.float32, np.float64])
def test_load_volume_accepts_numeric_dtypes_and_returns_uint8(tmp_path, dtype):
    cap = spherical_cap(8, math.radians(60))
    path = tmp_path / "vol.npy"
    np.save(path, cap.volume.astype(dtype))
    vol = load_volume(path)
    assert vol.dtype == np.uint8
    assert vol.flags.c_contiguous
    np.testing.assert_array_equal(vol, cap.volume)


@pytest.mark.parametrize(
    "bad_value", [3, -1, 0.5, 1.0000001, np.nan, np.inf], ids=lambda v: repr(v)
)
def test_validate_volume_rejects_values_outside_phase_labels(bad_value):
    vol = np.zeros((4, 4, 4), dtype=np.float64)
    vol[1, 2, 3] = bad_value
    with pytest.raises(ValueError, match="invalid labels"):
        validate_volume(vol)


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5, 5), (2, 5, 5), (5, 5, 1)])
def test_validate_volume_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        validate_volume(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize("dtype", [bool, complex, object])
def test_validate_volume_rejects_non_numeric_dtypes(dtype):
    with pytest.raises(ValueError, match="dtype"):
        validate_volume(np.zeros((4, 4, 4), dtype=dtype))


def test_load_volume_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_volume(tmp_path / "nope.npy")


def test_load_volume_rejects_object_array_and_npz(tmp_path):
    obj = tmp_path / "obj.npy"
    np.save(obj, np.array([[[{"a": 1}]]], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        load_volume(obj)
    npz = tmp_path / "vol.npz"
    np.savez(npz, a=np.zeros((4, 4, 4)))
    with pytest.raises(ValueError):
        load_volume(npz)


def test_phase_fractions_and_has_oil():
    vol = np.full((4, 5, 5), WATER, dtype=np.uint8)
    vol[0] = SOLID
    vol[1, 0, 0] = OIL
    fr = phase_fractions(vol)
    assert fr[SOLID] == pytest.approx(25 / 100)
    assert fr[OIL] == pytest.approx(1 / 100)
    assert sum(fr.values()) == pytest.approx(1.0)
    assert has_oil(vol)
    vol[vol == OIL] = WATER
    assert not has_oil(vol)


# ------------------------------------------------------------------ droplets.csv


def _row(**over):
    row = {
        "id": 1,
        "phase": "oil",
        "volume_vox": 1234,
        "centroid_z": 10.5,
        "centroid_y": 20.25,
        "centroid_x": 30.125,
        "touches_border": False,
        "n_loops": 1,
        "n_junctions": 0,
        "euler_surface": 2,
        "euler_droplet": 1.0,
        "area_lv": 1000.0 / 3,
        "area_ls": 200.0,
        "area_lv_um2": 1000.0 / 3 * 4.2**2,
        "area_ls_um2": 200.0 * 4.2**2,
        "sum_kappa_g_area_lv": 2 * math.pi * (1 + math.cos(math.pi / 4)),
        "sum_kappa_g_area_ls": 0.0,
        "deficit_curvature": 2 * math.pi * (1 - math.cos(math.pi / 4)),
        "deficit_curvature_raw": 2 * math.pi * (1 - math.cos(math.pi / 4)),
        "theta_macro_deg": 26.36,
        "theta_macro_raw_deg": 26.36,
        "status": "ok",
    }
    row.update(over)
    return row


def test_droplets_csv_round_trip_preserves_values_and_types(tmp_path):
    rows = [
        _row(),
        _row(id=2, touches_border=True, status="border", theta_macro_deg=None),
        _row(id=3, n_loops=0, status="no_contact", deficit_curvature=None, theta_macro_deg=None),
    ]
    path = write_droplets_csv(rows, tmp_path / "sub" / "droplets.csv")
    back = read_droplets_csv(path)
    assert len(back) == 3
    for orig, got in zip(rows, back):
        assert list(got) == list(DROPLET_COLUMNS)
        for col, kind in DROPLET_COLUMNS.items():
            assert got[col] == orig[col]  # float via repr is exact
            if got[col] is not None:
                assert type(got[col]) is kind


def test_droplets_csv_header_order_is_fixed(tmp_path):
    path = write_droplets_csv([], tmp_path / "d.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(DROPLET_COLUMNS)


def test_droplets_csv_accepts_numpy_scalars(tmp_path):
    row = _row(id=np.int64(7), volume_vox=np.int32(5), touches_border=np.bool_(True),
               area_lv=np.float32(1.5))
    back = read_droplets_csv(write_droplets_csv([row], tmp_path / "d.csv"))[0]
    assert back["id"] == 7 and back["touches_border"] is True and back["area_lv"] == 1.5


@pytest.mark.parametrize(
    ("over", "exc"),
    [
        ({"theta_macro": 1.0}, KeyError),  # typo in a column name
        ({"status": "fine"}, ValueError),
        ({"phase": "gas"}, ValueError),
        ({"n_loops": 1.5}, TypeError),
        ({"touches_border": 1}, TypeError),
    ],
)
def test_write_droplets_csv_rejects_bad_rows(tmp_path, over, exc):
    with pytest.raises(exc):
        write_droplets_csv([_row(**over)], tmp_path / "d.csv")


# ------------------------------------------------------------------ summary.json


def test_summarize_droplets_uses_only_ok_rows():
    rows = [
        _row(theta_macro_deg=40.0),
        _row(theta_macro_deg=50.0),
        _row(theta_macro_deg=60.0),
        _row(status="border", theta_macro_deg=None),
        _row(status="out_of_range", theta_macro_deg=190.0),
    ]
    s = summarize_droplets(rows)
    assert s["n_droplets_total"] == 5
    assert s["n_droplets_ok"] == 3
    assert s["theta_mean_deg"] == pytest.approx(50.0)
    assert s["theta_std_deg"] == pytest.approx(10.0)
    assert s["theta_median_deg"] == pytest.approx(50.0)
    assert s["status_counts"]["border"] == 1
    assert s["status_counts"]["out_of_range"] == 1


def test_summarize_droplets_handles_empty_and_single():
    empty = summarize_droplets([])
    assert empty["n_droplets_ok"] == 0 and empty["theta_mean_deg"] is None
    one = summarize_droplets([_row(theta_macro_deg=45.0)])
    assert one["theta_mean_deg"] == 45.0 and one["theta_std_deg"] is None


def test_summary_json_round_trip_with_numpy_and_nan(tmp_path):
    summary = {
        "n_droplets_ok": np.int64(3),
        "theta_mean_deg": np.float64(45.5),
        "theta_std_deg": float("nan"),
        "params": {"voxel_size_um": 4.2, "n_layer": np.int32(10), "shape": (300, 300, 300)},
        "input": tmp_path / "vol.npy",
        "note": "θ_macro, °",
    }
    path = write_summary_json(summary, tmp_path / "out" / "summary.json")
    back = read_summary_json(path)
    assert back["n_droplets_ok"] == 3
    assert back["theta_mean_deg"] == 45.5
    assert back["theta_std_deg"] is None
    assert back["params"]["shape"] == [300, 300, 300]
    assert back["note"] == "θ_macro, °"
    json.loads(path.read_text(encoding="utf-8"))  # valid JSON without NaN


# ------------------------------------------------------------------ real data


@pytest.mark.slow
def test_load_robuglas_45(data_dir):
    path = data_dir / "robuglas_45.npy"
    if not path.exists():
        pytest.skip(f"missing file {path}")
    vol = load_volume(path)
    assert vol.shape == (300, 300, 300)
    assert vol.dtype == np.uint8
    fr = phase_fractions(vol)
    assert fr[SOLID] == pytest.approx(0.6347, abs=1e-3)
    assert fr[WATER] == pytest.approx(0.2061, abs=1e-3)
    assert fr[OIL] == pytest.approx(0.1592, abs=1e-3)
