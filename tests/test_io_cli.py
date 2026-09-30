import json

import numpy as np
import pytest
import xarray as xr

from progress_kinetics import FitConfig, fit_dataset
from progress_kinetics.cli import load_dataset, main
from progress_kinetics.results import safe_dataset


@pytest.mark.parametrize("format", ["netcdf", "zarr"])
def test_result_round_trip_and_serializable_metadata(favorable_fit, tmp_path, format):
    favorable_fit.predictions["object_annotation"] = xr.DataArray(
        np.array([{"note": "synthetic"}], dtype=object), dims="plate_read"
    )
    favorable_fit.predictions.attrs["complex_annotation"] = {"source": "synthetic", "valid": True}
    root = tmp_path / format
    favorable_fit.write(root, format=format, plots=False)
    suffix = ".nc" if format == "netcdf" else ".zarr"
    ds = load_dataset(root / ("fit_results" + suffix))
    assert ds.sizes["well_384"] == 12
    assert ds.prediction_AU.dims == ("plate_read", "time", "well_384")
    np.testing.assert_allclose(ds.prediction_AU, favorable_fit.predictions.prediction_AU)
    assert ds.object_annotation.dtype.kind in "OUS"
    report = json.loads((root / "run.json").read_text())
    assert report["exit_code"] == 0
    assert report["versions"]["torch"]
    assert (root / "fit_parameters.csv").exists()
    assert (root / "trace_diagnostics.csv").exists()
    assert (root / "config.json").exists()
    assert "group" in ds.dims and "trace_baseline_AU" in ds


def test_plot_export_and_original_input_retained(favorable_fit, tmp_path):
    favorable_fit.write(tmp_path, plots=True)
    files = list((tmp_path / "plots").glob("*.png"))
    assert len(files) >= 4 and all(f.stat().st_size > 1000 for f in files)
    ds = load_dataset(tmp_path / "fit_results.nc")
    assert "pathlength" in ds and "PLASMIDID" in ds
    assert "original_sample_index" in ds.coords


@pytest.mark.parametrize("format", ["netcdf", "zarr"])
def test_cli_api_parity(single, tmp_path, capsys, format):
    path = tmp_path / ("input.nc" if format == "netcdf" else "input.zarr")
    if format == "netcdf":
        safe_dataset(single).to_netcdf(path, engine="h5netcdf")
    else:
        safe_dataset(single).to_zarr(path, zarr_format=2)
    c = FitConfig(adam_steps=20, lbfgs_steps=80, multistart=1, plots=False)
    config = tmp_path / "config.json"
    c.to_json(config)
    out = tmp_path / "results"
    assert (
        main([str(path), "--config", str(config), "--outdir", str(out), "--output-format", format])
        == 0
    )
    content = json.loads(capsys.readouterr().out)
    api = fit_dataset(single, c)
    assert content["groups"][0]["kcat_per_s"] == pytest.approx(
        api.parameter_rows[0]["kcat_per_s"], rel=1e-10
    )
    run = json.loads((out / "run.json").read_text())
    assert run["config"]["adam_steps"] == 20
    assert run["input_path"] == str(path.resolve())


def test_cli_validation_invalid_and_partial_exit_policy(example, tmp_path, capsys):
    path = tmp_path / "input.nc"
    safe_dataset(example).to_netcdf(path, engine="h5netcdf")
    report_path = tmp_path / "inspect.json"
    assert main(["validate", str(path), "--report-json", str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    assert len(report["groups"]) == 4 and report["conventions"]["signal_mode"] == "corrected"
    assert not (tmp_path / "results").exists()
    capsys.readouterr()
    assert (
        main([str(path), "--extinction-coefficient-M-cm", "6000", "--fix-alpha-AU-per-mM", "6"])
        == 2
    )
    assert "mutually exclusive" in json.loads(capsys.readouterr().err)["error"]
    assert main([str(tmp_path / "missing.nc")]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "invalid_input"
    ds = example.copy(deep=True)
    ds.absorbance_corrected.values[:, :, 12:24] = 1
    safe_dataset(ds).to_netcdf(path, engine="h5netcdf", mode="w")
    out = tmp_path / "partial"
    assert (
        main(
            [
                str(path),
                "--merge-plates",
                "--adam-steps",
                "10",
                "--lbfgs-steps",
                "80",
                "--multistart",
                "1",
                "--no-plots",
                "--outdir",
                str(out),
            ]
        )
        == 3
    )
    assert json.loads(capsys.readouterr().out)["status"] == "partial_success"
    ds.absorbance_corrected.values[:] = 1
    safe_dataset(ds).to_netcdf(path, engine="h5netcdf", mode="w")
    assert main([str(path), "--no-plots", "--outdir", str(tmp_path / "none")]) == 4
    assert json.loads(capsys.readouterr().out)["status"] == "no_valid_fit"


def test_cli_config_override_and_example(tmp_path, capsys):
    path = tmp_path / "example.nc"
    assert main(["example", str(path), "--seed", "12"]) == 0
    capsys.readouterr()
    c = FitConfig(adam_steps=0, lbfgs_steps=0, multistart=1, plots=False)
    config = tmp_path / "config.json"
    c.to_json(config)
    out = tmp_path / "override"
    assert (
        main(
            [
                str(path),
                "--config",
                str(config),
                "--outdir",
                str(out),
                "--merge-plates",
                "--start",
                "kcat_per_s=12",
                "--select",
                "PLASMIDID=SYNTH_A",
            ]
        )
        == 3
    )
    report = json.loads((out / "run.json").read_text())
    assert report["config"]["starts"]["kcat_per_s"] == 12
    assert report["config"]["starts"]["Km_mM"] == 0.3
    assert report["config"]["adam_steps"] == 0 and report["config"]["merge_plates"]
    assert any(r["status"] == "no_kinetic_traces" for r in report["group_statuses"])


def test_saturation_out_of_fit_and_missing_output(single):
    ds = single.copy(deep=True)
    ds.saturated.values[0, 60, 11] = True
    ds.absorbance_corrected.values[0, 2, 0] = np.nan
    result = fit_dataset(ds, FitConfig(adam_steps=20, lbfgs_steps=80, multistart=1, plots=False))
    assert not result.predictions.used_for_fit.values[0, 60:, 11].any()
    assert np.isfinite(result.predictions.prediction_AU.values[0, 60:, 11]).all()
    assert np.isnan(result.predictions.residual_AU.values[0, 60:, 11]).all()
    assert np.isnan(result.predictions.observed_AU.values[0, 2, 0])
    xr.testing.assert_identical(
        ds, result.predictions[list(ds.data_vars)].drop_vars("original_sample_index")
    )


def test_json_config_roundtrip_and_reject_bad_configuration(tmp_path):
    config = FitConfig(reaction_age_at_first_read_s={"plate_1": 3}, selections={"flag": [True]})
    path = tmp_path / "settings.json"
    config.to_json(path)
    restored = FitConfig.from_json(path)
    assert restored.to_json() == config.to_json()
    assert restored.reaction_age_at_first_read_s == {"plate_1": 3}


@pytest.mark.parametrize(
    "options",
    [
        {"minimum_points": 2},
        {"reporter_stoichiometry": 0},
        {"signal_direction": "sideways"},
        {"reaction_age_at_first_read_s": -1},
        {"device": "mps"},
        {"lag_s": -1},
        {"signal_mode": "raw", "fix_alpha_AU_per_mM": 6},
        {"group_by": ("cofactor_concentration_mM",)},
        {"fixed_parameters": {"mystery": 3}},
        {"multistart": 0},
        {"extinction_coefficient_M_cm": float("nan")},
        {"duplicate_time": "ignore"},
        {"direction_agreement": 0.4},
    ],
)
def test_invalid_config(options):
    with pytest.raises(ValueError):
        FitConfig(**options)


@pytest.mark.parametrize("options", [{"seed": -1}, {"multistart": 1.5}, {"adam_steps": True}])
def test_invalid_optimizer_count_types(options):
    with pytest.raises(ValueError):
        FitConfig(**options)
