import math

import numpy as np
import pytest

from gauss_bonnet.cli import (
    DROPLETS_CSV,
    SUMMARY_JSON,
    Params,
    analyze_phases,
    analyze_volume,
    main,
    run,
    sample_droplets,
)
from gauss_bonnet.droplets import find_droplets
from gauss_bonnet.io import (
    STATUS_BORDER,
    STATUS_ERROR,
    STATUS_NOT_SAMPLED,
    STATUS_OK,
    STATUS_TOO_SMALL,
    read_droplets_csv,
    read_summary_json,
    summarize_droplets,
)
from gauss_bonnet.phases import OIL, SOLID, swap_fluids
from gauss_bonnet.synthetic import spherical_cap


def _scene():
    """Two droplets on a substrate, a droplet at the image edge and a single oil voxel."""
    cap = spherical_cap(10, math.radians(90)).volume
    vol = np.concatenate([cap, cap], axis=2)
    vol[cap.shape[0] // 2 : cap.shape[0] // 2 + 3, 0:3, 0:3] = OIL  # at the edge: border
    vol[3, -3, cap.shape[2]] = OIL  # single voxel on the solid: too_small
    assert vol[2, -3, cap.shape[2]] == SOLID
    return vol


def test_analyze_volume_assigns_statuses_and_angles():
    rows = analyze_volume(_scene())
    statuses = sorted(r["status"] for r in rows)
    assert statuses == sorted([STATUS_OK, STATUS_OK, STATUS_BORDER, STATUS_TOO_SMALL])
    for r in rows:
        if r["status"] == STATUS_OK:
            assert r["theta_macro_deg"] == pytest.approx(90, abs=6)
            assert r["n_loops"] == 1 and r["euler_surface"] == 2
        else:
            assert "theta_macro_deg" not in r


def test_params_are_passed_through():
    vol = spherical_cap(10, math.radians(90)).volume
    a = analyze_volume(vol)[0]
    b = analyze_volume(vol, Params(euler="ball", smooth_iters=0, curvature_method="quadric"))[0]
    assert a["theta_macro_deg"] != b["theta_macro_deg"]
    c = analyze_volume(vol, Params(min_droplet_voxels=10**9))[0]
    assert c["status"] == STATUS_TOO_SMALL


def test_run_writes_csv_and_summary(tmp_path):
    vol = _scene()
    np.save(tmp_path / "scene.npy", vol.astype(np.float64))  # float64, like robuglas_45
    summary = run(tmp_path / "scene.npy", tmp_path / "out", voxel_size=4.2)
    rows = read_droplets_csv(tmp_path / "out" / DROPLETS_CSV)
    saved = read_summary_json(tmp_path / "out" / SUMMARY_JSON)
    assert len(rows) == saved["n_droplets_total"] == 4
    assert saved["n_droplets_ok"] == 2
    assert saved["theta_median_deg"] == pytest.approx(90, abs=6)
    assert saved["voxel_size_um"] == 4.2 and saved["params"]["euler"] == "mesh"
    assert saved["shape"] == list(vol.shape)
    assert summary["n_droplets_ok"] == saved["n_droplets_ok"]
    ok = [r for r in rows if r["status"] == STATUS_OK]
    assert ok[0]["area_lv_um2"] == pytest.approx(ok[0]["area_lv"] * 4.2**2)


def test_main_cli_round_trip(tmp_path, capsys):
    np.save(tmp_path / "cap.npy", spherical_cap(8, math.radians(90)).volume)
    code = main([str(tmp_path / "cap.npy"), "--out", str(tmp_path / "res"),
                 "--sign-correction", "true", "--n-layer", "2"])
    assert code == 0
    assert "θ_macro" in capsys.readouterr().out
    saved = read_summary_json(tmp_path / "res" / SUMMARY_JSON)
    assert saved["params"]["sign_correction"] is True and saved["params"]["n_layer"] == 2


def test_main_reports_bad_input(tmp_path, capsys):
    np.save(tmp_path / "bad.npy", np.full((5, 5, 5), 7))
    assert main([str(tmp_path / "bad.npy"), "--out", str(tmp_path / "r"), "--quiet"]) == 2
    assert "invalid labels" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.npy"), "--quiet"]) == 2


def test_main_rejects_bad_bool(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "x.npy"), "--triple-line", "maybe"])


@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [({"n_layer": 0}, "--n-layer"), ({"smooth_alpha": 1.5}, "--smooth-alpha"),
     ({"euler": "torus"}, "--euler"), ({"min_droplet_voxels": 0}, "--min-droplet-voxels"),
     ({"droplet_phase": "gas"}, "--droplet-phase"), ({"max_droplets": 0}, "--max-droplets"),
     ({"max_droplets": 2.5}, "--max-droplets"), ({"seed": -1}, "--seed")],
)
def test_params_reject_bad_values_naming_the_flag(kwargs, flag):
    with pytest.raises(ValueError, match=flag):
        Params(**kwargs)


@pytest.mark.parametrize("argv", [["--n-layer", "0"], ["--voxel-size", "0"],
                                  ["--voxel-size", "abc"], ["--smooth-iters", "-1"]])
def test_main_rejects_bad_parameters_before_loading(tmp_path, capsys, argv):
    # the file is missing: if parameters were checked after loading, the error would be about the file
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "missing.npy"), *argv])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not found" not in err and argv[0] in err


def test_failing_droplet_gets_error_status_and_others_continue(monkeypatch, caplog):
    from gauss_bonnet import cli

    real = cli.face_curvature
    calls = []

    def flaky(mesh, *a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise FloatingPointError("test failure")
        return real(mesh, *a, **kw)

    monkeypatch.setattr(cli, "face_curvature", flaky)
    with caplog.at_level("WARNING", logger="gauss_bonnet"):
        rows = analyze_volume(_scene())
    statuses = sorted(r["status"] for r in rows)
    assert statuses == sorted([STATUS_ERROR, STATUS_OK, STATUS_BORDER, STATUS_TOO_SMALL])
    assert "test failure" in caplog.text and "FloatingPointError" in caplog.text
    summary = summarize_droplets(rows)
    assert summary["status_counts"][STATUS_ERROR] == 1 and summary["n_droplets_ok"] == 1


def test_main_writes_log_file_and_quiet_prints_nothing(tmp_path, capsys):
    np.save(tmp_path / "cap.npy", spherical_cap(8, math.radians(90)).volume)
    log_file = tmp_path / "logs" / "run.log"
    code = main([str(tmp_path / "cap.npy"), "--out", str(tmp_path / "res"), "-q",
                 "--log-file", str(log_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    text = log_file.read_text(encoding="utf-8")
    assert "loading" in text and "droplet oil (label 2) 1:" in text and "DEBUG" in text


def test_main_reports_out_path_that_is_a_file(tmp_path, capsys):
    np.save(tmp_path / "cap.npy", spherical_cap(6, math.radians(90)).volume)
    (tmp_path / "taken").write_text("x")
    assert main([str(tmp_path / "cap.npy"), "--out", str(tmp_path / "taken")]) == 2
    assert "points to a file" in capsys.readouterr().err


def test_volume_without_oil_warns(tmp_path, capsys):
    vol = np.full((6, 6, 6), 1, dtype=np.uint8)
    vol[:2] = SOLID
    np.save(tmp_path / "dry.npy", vol)
    assert main([str(tmp_path / "dry.npy"), "--out", str(tmp_path / "r")]) == 0
    captured = capsys.readouterr()
    assert "no oil" in captured.err and "(none)" in captured.out
    assert read_droplets_csv(tmp_path / "r" / DROPLETS_CSV) == []


@pytest.mark.slow
def test_robuglas_45_median_angle(data_dir, tmp_path):
    """Integration test: true angle 45°, the θ_macro median is compared."""
    path = data_dir / "robuglas_45.npy"
    if not path.exists():
        pytest.skip(f"missing file {path}")
    summary = run(path, tmp_path)
    assert summary["n_droplets_ok"] >= 100
    assert summary["theta_median_deg"] == pytest.approx(45, abs=10)


def test_water_droplet_angle_is_supplement_of_oil_droplet():
    """A water droplet in oil has the same geometry as an oil droplet in water: angle through water π − θ."""
    cap = spherical_cap(10, math.radians(60)).volume
    oil = analyze_volume(cap, Params(droplet_phase="oil"))
    water = analyze_volume(swap_fluids(cap), Params(droplet_phase="water"))
    ok_oil = [r for r in oil if r["status"] == STATUS_OK]
    ok_water = [r for r in water if r["status"] == STATUS_OK]
    assert len(ok_oil) == len(ok_water) == 1
    assert ok_water[0]["phase"] == "water"
    assert ok_water[0]["theta_macro_deg"] == pytest.approx(180 - ok_oil[0]["theta_macro_deg"])


@pytest.mark.parametrize(("swap", "expected"), [(False, "oil"), (True, "water")])
def test_auto_mode_picks_phase_with_more_ok_droplets(swap, expected):
    cap = spherical_cap(10, math.radians(60)).volume
    vol = swap_fluids(cap) if swap else cap
    rows, sel = analyze_phases(vol, Params(droplet_phase="auto"))
    assert sel["mode"] == "auto" and sel["droplet_phase"] == expected
    assert set(sel["phases"]) == {"oil", "water"}
    assert sel["phases"][expected]["n_droplets_ok"] == 1
    assert all(r["phase"] == expected for r in rows)
    fixed, sel_fixed = analyze_phases(vol, Params(droplet_phase="oil"))
    assert sel_fixed["droplet_phase"] == "oil" and set(sel_fixed["phases"]) == {"oil"}
    assert all(r["phase"] == "oil" for r in fixed)


def test_default_droplet_phase_is_auto():
    assert Params().droplet_phase == "auto"
    _, sel = analyze_phases(swap_fluids(spherical_cap(8, math.radians(120)).volume))
    assert sel["mode"] == "auto" and set(sel["phases"]) == {"oil", "water"}
    assert sel["droplet_phase"] == "water"


def test_run_records_droplet_phase_in_summary(tmp_path, capsys):
    cap = spherical_cap(8, math.radians(120)).volume
    np.save(tmp_path / "cap.npy", swap_fluids(cap))
    assert main([str(tmp_path / "cap.npy"), "--out", str(tmp_path / "r"),
                 "--droplet-phase", "auto"]) == 0
    captured = capsys.readouterr()
    assert "droplets: water" in captured.out and "auto mode chose water droplets" in captured.err
    saved = read_summary_json(tmp_path / "r" / SUMMARY_JSON)
    assert saved["droplet_phase"] == "water" and saved["params"]["droplet_phase"] == "auto"
    assert saved["phase_selection"]["phases"]["water"]["n_droplets_ok"] == 1
    oil = [r for r in analyze_volume(cap, Params(droplet_phase="oil")) if r["status"] == STATUS_OK]
    assert saved["theta_median_deg"] == pytest.approx(180 - oil[0]["theta_macro_deg"])
    rows = read_droplets_csv(tmp_path / "r" / DROPLETS_CSV)
    assert {r["phase"] for r in rows} == {"water"}


@pytest.mark.slow
def test_robuglas_135_uses_water_droplets(data_dir, tmp_path):
    """Hydrophobic sample: almost all oil droplets touch the edge, auto takes water droplets.

    Currently the median is 115.7° over 196 ok droplets: the method underestimates 105-135° by 12-20°.
    """
    path = data_dir / "robuglas_135.npy"
    if not path.exists():
        pytest.skip(f"missing file {path}")
    summary = run(path, tmp_path, Params(droplet_phase="auto"))
    assert summary["droplet_phase"] == "water"
    assert summary["n_droplets_ok"] >= 150
    assert summary["theta_median_deg"] == pytest.approx(135, abs=25)


def _many_drops(n=6):
    """n identical hemispheres in a row on a substrate."""
    cap = spherical_cap(5, math.radians(90)).volume
    return np.concatenate([cap] * n, axis=2)


def test_sample_droplets_is_reproducible_subset():
    _, droplets = find_droplets(_many_drops())
    ok_ids = {d.id for d in droplets if d.status == STATUS_OK}
    assert len(ok_ids) == 6
    assert sample_droplets(droplets, None, 0, "oil") == ok_ids
    assert sample_droplets(droplets, 10, 0, "oil") == ok_ids
    a = sample_droplets(droplets, 3, 0, "oil")
    assert len(a) == 3 and a <= ok_ids
    assert a == sample_droplets(droplets, 3, 0, "oil")
    picks = {frozenset(sample_droplets(droplets, 3, s, "oil")) for s in range(20)}
    assert len(picks) > 1  # the seed changes the sample


def test_max_droplets_limits_measured_droplets():
    vol = _many_drops()
    rows, sel = analyze_phases(vol, Params(max_droplets=2, seed=7, droplet_phase="auto"))
    statuses = [r["status"] for r in rows]
    assert statuses.count(STATUS_OK) == 2 and statuses.count(STATUS_NOT_SAMPLED) == 4
    assert all("theta_macro_deg" not in r for r in rows if r["status"] == STATUS_NOT_SAMPLED)
    oil = sel["phases"]["oil"]
    assert oil["n_candidates"] == 6 and oil["n_sampled"] == 2
    assert oil["n_droplets_ok_estimated"] == pytest.approx(6)
    again, _ = analyze_phases(vol, Params(max_droplets=2, seed=7, droplet_phase="oil"))
    assert [r["status"] for r in again] == statuses  # auto and oil pick the same droplets


def test_main_max_droplets_flag(tmp_path, capsys):
    np.save(tmp_path / "many.npy", _many_drops())
    assert main([str(tmp_path / "many.npy"), "--out", str(tmp_path / "r"),
                 "--max-droplets", "3", "--seed", "1"]) == 0
    assert "not_sampled 3" in capsys.readouterr().out
    saved = read_summary_json(tmp_path / "r" / SUMMARY_JSON)
    assert saved["n_droplets_ok"] == 3 and saved["params"]["max_droplets"] == 3
    assert saved["status_counts"][STATUS_NOT_SAMPLED] == 3
