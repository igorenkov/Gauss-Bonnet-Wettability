import math

import numpy as np
import pytest

from gauss_bonnet.cli import DROPLETS_CSV, main, run
from gauss_bonnet.io import STATUS_BORDER, STATUS_OK, STATUS_OUT_OF_RANGE, write_droplets_csv
from gauss_bonnet.plots import (
    STATUSES_FILE,
    THETA_HIST_FILE,
    THETA_VS_VOLUME_FILE,
    plot_results,
    plot_status_counts,
    plot_theta_histogram,
    plot_theta_vs_volume,
    save_plots,
    theta_values,
)
from gauss_bonnet.plots import main as plot_main
from gauss_bonnet.synthetic import spherical_cap

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _rows():
    rng = np.random.default_rng(0)
    rows = []
    for i, theta in enumerate(rng.normal(50, 15, 40), start=1):
        rows.append({"id": i, "volume_vox": int(rng.integers(30, 5000)), "status": STATUS_OK,
                     "theta_macro_deg": float(theta), "n_loops": 1 + (i % 7 == 0),
                     "n_junctions": int(i % 11 == 0)})
    rows.append({"id": 41, "volume_vox": 100, "status": STATUS_OUT_OF_RANGE,
                 "theta_macro_deg": -12.0, "n_loops": 1, "n_junctions": 0})
    rows.append({"id": 42, "volume_vox": 90000, "status": STATUS_BORDER})
    return rows


def test_theta_values_takes_only_requested_status():
    rows = _rows()
    assert theta_values(rows).size == 40
    np.testing.assert_array_equal(theta_values(rows, STATUS_OUT_OF_RANGE), [-12.0])


def test_histogram_counts_all_ok_droplets_in_0_180():
    rows = _rows()
    fig = plot_theta_histogram(rows, bin_width_deg=10, true_angle_deg=45)
    ax = fig.axes[0]
    heights = [p.get_height() for p in ax.patches]
    assert len(heights) == 18 and sum(heights) == 40
    assert ax.get_xlim() == (0, 180)
    title = ax.get_title(loc="left")
    assert "ok droplets: 40" in title and "not shown: 1" in title
    labels = ax.get_legend_handles_labels()[1]
    assert any(lbl.startswith("median") for lbl in labels)
    assert any("true angle 45" in lbl for lbl in labels)


def test_histogram_with_uneven_bin_width_ends_at_180():
    fig = plot_theta_histogram(_rows(), bin_width_deg=7)
    edges = [p.get_x() + p.get_width() for p in fig.axes[0].patches]
    assert max(edges) == pytest.approx(180)


def test_scatter_splits_contact_line_groups():
    fig = plot_theta_vs_volume(_rows())
    ax = fig.axes[0]
    sizes = [len(c.get_offsets()) for c in ax.collections]
    assert sum(sizes) == 40 and len(sizes) == 3
    assert ax.get_xscale() == "log"


def test_status_plot_shows_counts_and_oil_share():
    fig = plot_status_counts(_rows())
    ax_n, ax_v = fig.axes
    assert sorted(p.get_width() for p in ax_n.patches) == [1, 1, 40]
    assert sum(p.get_width() for p in ax_v.patches) == pytest.approx(100)


@pytest.mark.parametrize("rows", [[], [{"id": 1, "volume_vox": 50, "status": STATUS_BORDER}]])
def test_plots_survive_no_ok_droplets(tmp_path, rows):
    paths = save_plots(rows, tmp_path)
    assert [p.name for p in paths] == [f"{THETA_HIST_FILE}.png", f"{THETA_VS_VOLUME_FILE}.png",
                                       f"{STATUSES_FILE}.png"]
    for p in paths:
        assert p.read_bytes()[:8] == PNG_MAGIC


@pytest.mark.parametrize("kwargs", [{"bin_width_deg": 0}, {"true_angle_deg": 200},
                                    {"fmt": "bmp"}])
def test_save_plots_rejects_bad_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        save_plots(_rows(), tmp_path, **kwargs)


def test_plot_results_reads_csv_and_writes_svg(tmp_path):
    write_droplets_csv(_rows(), tmp_path / DROPLETS_CSV)
    paths = plot_results(tmp_path, tmp_path / "fig", fmt="svg")
    assert all(p.parent == tmp_path / "fig" and p.suffix == ".svg" for p in paths)
    assert paths[0].read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_plot_cli_reports_missing_results(tmp_path, capsys):
    assert plot_main([str(tmp_path / "nothing"), "-q"]) == 2
    assert "run gauss-bonnet" in capsys.readouterr().err


def test_plot_cli_on_real_run(tmp_path, capsys):
    np.save(tmp_path / "cap.npy", spherical_cap(8, math.radians(90)).volume)
    run(tmp_path / "cap.npy", tmp_path / "res")
    assert plot_main([str(tmp_path / "res"), "--true-angle", "90"]) == 0
    out = capsys.readouterr().out
    assert f"{THETA_HIST_FILE}.png" in out
    assert (tmp_path / "res" / f"{STATUSES_FILE}.png").is_file()


def test_main_plots_flag(tmp_path, capsys):
    np.save(tmp_path / "cap.npy", spherical_cap(8, math.radians(90)).volume)
    code = main([str(tmp_path / "cap.npy"), "--out", str(tmp_path / "res"), "--plots",
                 "--true-angle", "90"])
    assert code == 0
    assert "plots:" in capsys.readouterr().out
    assert (tmp_path / "res" / f"{THETA_HIST_FILE}.png").is_file()


@pytest.mark.parametrize("argv", [["--true-angle", "45"], ["--plots", "--true-angle", "500"]])
def test_main_rejects_bad_true_angle_before_run(tmp_path, capsys, argv):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "missing.npy"), *argv])
    assert exc.value.code == 2 and "--true-angle" in capsys.readouterr().err
