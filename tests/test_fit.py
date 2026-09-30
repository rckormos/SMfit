from dataclasses import replace

import numpy as np
import pytest
import torch
import xarray as xr

from progress_kinetics import FitConfig, fit_dataset
from progress_kinetics.data import prepare_dataset
from progress_kinetics.direction import detect_direction
from progress_kinetics.model import make_batch
from progress_kinetics.optimize import optimize
from progress_kinetics.synthetic import ode_substrate

FAST = FitConfig(adam_steps=25, lbfgs_steps=100, multistart=1, plots=False)


def assert_recovery(result, km=0.3, kcat=12, epsilon=6000, tolerance=0.01):
    row = result.parameter_rows[0]
    assert row["status"] == "success"
    assert row["Km_mM"] == pytest.approx(km, rel=tolerance)
    assert row["kcat_per_s"] == pytest.approx(kcat, rel=tolerance)
    if epsilon is not None:
        assert row["epsilon_per_M_cm"] == pytest.approx(epsilon, rel=tolerance)
    return row


def test_fitted_optical_recovery_independent_ode(favorable_fit):
    row = assert_recovery(favorable_fit, tolerance=0.02)
    assert row["loss"] < 1e-9
    assert not row["weak_identification"]
    assert favorable_fit.exit_code == 0


def test_fixed_optical_noise_free_recovery(single):
    result = fit_dataset(single, replace(FAST, extinction_coefficient_M_cm=6000))
    assert_recovery(result, tolerance=0.01)
    assert result.parameter_rows[0]["optical_fixed"]


def test_corrected_ignores_measured_lengths_and_stoichiometry(single):
    ds = single.copy(deep=True)
    ds.pathlength.values[:] = 0.001
    result = fit_dataset(ds, replace(FAST, reporter_stoichiometry=2))
    assert_recovery(result, epsilon=3000, tolerance=0.02)
    assert result.parameter_rows[0]["alpha_AU_per_mM"] == pytest.approx(6, rel=0.02)
    result = fit_dataset(ds, replace(FAST, reference_path_length_cm=0.5))
    assert_recovery(result, epsilon=12000, tolerance=0.02)


def test_raw_varying_lengths_independent_scale(single):
    ds = single.copy(deep=True)
    length = ds.pathlength.values[:, None, :]
    ds.absorbance_corrected.values[:] = 20 + length * (ds.absorbance_corrected.values - 20)
    result = fit_dataset(ds, replace(FAST, signal_mode="raw", extinction_coefficient_M_cm=6000))
    assert_recovery(result, tolerance=0.01)
    for trace in result.trace_rows:
        assert trace["alpha_AU_per_mM"] == pytest.approx(6 * trace["length_cm"])


def test_fixed_alpha_and_effective_alpha_only(single):
    result = fit_dataset(single, replace(FAST, fix_alpha_AU_per_mM=6))
    assert_recovery(result, epsilon=None)
    assert np.isnan(result.parameter_rows[0]["epsilon_per_M_cm"])
    assert result.run["conventions"]["optical_parameter"] == "alpha"
    result = fit_dataset(
        single.drop_vars("pathlength"), replace(FAST, signal_mode="raw", effective_alpha_only=True)
    )
    assert_recovery(result, epsilon=None, tolerance=0.02)
    assert result.parameter_rows[0]["alpha_AU_per_mM"] == pytest.approx(6, rel=0.02)


def test_both_signs_multiple_groups_and_reproducibility(example):
    before = example.copy(deep=True)
    result = fit_dataset(example, replace(FAST, merge_plates=True))
    assert len(result.parameter_rows) == 2
    assert_recovery(result, tolerance=0.02)
    second = result.parameter_rows[1]
    assert second["Km_mM"] == pytest.approx(0.6, rel=0.02)
    assert second["kcat_per_s"] == pytest.approx(20, rel=0.02)
    assert second["direction"] == "increasing"
    assert len(result.trace_rows) == 52
    xr.testing.assert_identical(example, before)
    ds = example.isel(plate_read=[0], well_384=slice(0, 12))
    a = fit_dataset(ds, FAST)
    b = fit_dataset(ds, FAST)
    assert a.parameter_rows == b.parameter_rows
    xr.testing.assert_identical(a.predictions, b.predictions)


def test_majority_conflicting_flat_too_few_and_override(single):
    ds = single.copy(deep=True)
    ds.absorbance_corrected.values[:, :, 9:] = 40 - ds.absorbance_corrected.values[:, :, 9:]
    data = prepare_dataset(ds, FitConfig())
    direction, report = detect_direction(data, list(range(12)))
    assert direction == -1 and report["source"] == "auto"
    assert report["agreement"] >= 0.75
    ds.absorbance_corrected.values[:, :, 6:9] = 40 - ds.absorbance_corrected.values[:, :, 6:9]
    result = fit_dataset(ds, FAST)
    assert result.parameter_rows[0]["status"] == "ambiguous_direction"
    assert result.exit_code == 4 and not result.predictions.used_for_fit.any()
    small = single.isel(well_384=[6, 7])
    assert fit_dataset(small, FAST).exit_code == 4
    override = fit_dataset(small, replace(FAST, signal_direction="decreasing"))
    assert override.exit_code == 0
    assert next(iter(override.run["group_reports"].values()))["direction"]["source"] == "override"
    flat = single.copy(deep=True)
    flat.absorbance_corrected.values[:] = 1
    assert fit_dataset(flat, FAST).parameter_rows[0]["status"] == "ambiguous_direction"
    noisy = single.copy(deep=True)
    noisy.absorbance_corrected.values[:] = np.random.default_rng(2).normal(
        1, 1, noisy.absorbance_corrected.shape
    )
    d, r = detect_direction(prepare_dataset(noisy, FAST), list(range(12)))
    assert d is None and r["flat_or_noisy"] > 0


def test_control_traces_do_not_vote(example):
    data = prepare_dataset(example, FAST)
    group = next(iter(data.groups.values()))
    indices = [i for i in group if data.traces[i]["status"] == "ready"]
    _, report = detect_direction(data, indices)
    assert len(report["trace_votes"]) == 12


def test_reaction_age_missing_early_and_lag(single):
    ds = single.copy(deep=True)
    age, lag = 35, 10
    for i in range(ds.sizes["well_384"]):
        s0 = float(ds.cofactor_concentration_mM.values[i, 0])
        e = float(ds.expected_conc_um_NORM1.values[0, i])
        s = ode_substrate(np.maximum(ds.time.values + age - lag, 0), s0, e, 0.3, 12)
        ds.absorbance_corrected.values[0, :, i] = 20 - 6 * (s0 - s)
    ds.absorbance_corrected.values[0, :7, 0] = np.nan
    c = replace(FAST, reaction_age_at_first_read_s=age, lag_s=lag, extinction_coefficient_M_cm=6000)
    result = fit_dataset(ds, c)
    assert_recovery(result, tolerance=0.01)
    assert result.predictions.reaction_time_s.values[0, 7, 0] == 175
    # Positive fitted lag is a separate option; reaction age remains externally fixed.
    ds = single.copy(deep=True)
    for i in range(ds.sizes["well_384"]):
        s0 = float(ds.cofactor_concentration_mM.values[i, 0])
        e = float(ds.expected_conc_um_NORM1.values[0, i])
        s = ode_substrate(np.maximum(ds.time.values - 30, 0), s0, e, 0.3, 12)
        ds.absorbance_corrected.values[0, :, i] = 20 - 6 * (s0 - s)
    c = replace(
        FAST, fit_lag=True, extinction_coefficient_M_cm=6000, starts={**FAST.starts, "lag_s": 20}
    )
    result = fit_dataset(ds, c)
    assert_recovery(result, tolerance=0.02)
    assert result.parameter_rows[0]["lag_s"] == pytest.approx(30, abs=0.5)


def test_low_depletion_and_narrow_range_flags(single):
    low = single.isel(time=slice(0, 6)).copy(deep=True)
    low.expected_conc_um_NORM1.values[:] *= 0.001
    for i in range(low.sizes["well_384"]):
        s0 = float(low.cofactor_concentration_mM.values[i, 0])
        e = float(low.expected_conc_um_NORM1.values[0, i])
        s = ode_substrate(low.time.values, s0, e, 0.3, 12)
        low.absorbance_corrected.values[0, :, i] = 20 - 6 * (s0 - s)
    result = fit_dataset(low, replace(FAST, signal_direction="decreasing"))
    assert result.parameter_rows[0]["weak_identification"]
    assert (
        "low_depletion_optical_kcat_confounding"
        in result.parameter_rows[0]["identifiability_reasons"]
    )
    narrow = single.isel(well_384=[6, 7])
    result = fit_dataset(narrow, replace(FAST, signal_direction="decreasing"))
    assert (
        "narrow_substrate_range_Km_recovery_uncertain"
        in result.parameter_rows[0]["identifiability_reasons"]
    )


def test_profile_reoptimizes_and_has_no_intervals(single):
    result = fit_dataset(
        single, replace(FAST, identifiability="profile", profile_factors=(0.5, 1, 2))
    )
    report = next(iter(result.run["group_reports"].values()))["identifiability"]
    assert len(report["optical_profile"]) == 3
    assert all("parameters" in row for row in report["optical_profile"])
    assert "not_confidence_intervals" in report["intervals"]
    assert report["optical_profile"][1]["loss"] < 1e-8


def test_history_loss_values_and_best_state(single):
    data = prepare_dataset(single, FAST)
    batch = make_batch(data, list(range(12)), -1)
    model, report = optimize(FAST, batch, FAST.starts)
    assert report["loss"] == min(h["loss"] for h in report["history"])
    for history in report["history"][:: max(1, len(report["history"]) // 10)]:
        physical = {key: torch.tensor(history[key], dtype=torch.float64) for key in FAST.bounds}
        loss = float(model.evaluate(batch, physical)[0])
        assert loss == pytest.approx(history["loss"], abs=1e-13)


def test_nonfinite_trial_restores_best_finite_state(single, monkeypatch):
    from progress_kinetics.model import GroupModel

    original = GroupModel.evaluate
    calls = 0

    def failing(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(self, *args, **kwargs)
        if calls == 3:
            return (result[0] * float("nan"), *result[1:])
        return result

    monkeypatch.setattr(GroupModel, "evaluate", failing)
    data = prepare_dataset(single, FAST)
    model, report = optimize(FAST, make_batch(data, list(range(12)), -1), FAST.starts)
    assert model is not None
    assert report["status"] == "best_finite_restored"
    assert np.isfinite(report["loss"])


def test_bound_hits_and_no_optimizer_budget(single):
    c = replace(
        FAST,
        extinction_coefficient_M_cm=6000,
        bounds={**FAST.bounds, "kcat_per_s": (0.1, 11)},
        starts={**FAST.starts, "kcat_per_s": 10},
    )
    result = fit_dataset(single, c)
    assert "kcat_per_s" in result.parameter_rows[0]["bound_hits"]
    result = fit_dataset(single, replace(FAST, adam_steps=0, lbfgs_steps=0))
    assert result.parameter_rows[0]["status"] == "success"
    assert not result.parameter_rows[0]["converged"]


def test_shared_drift_opt_in(single):
    ds = single.copy(deep=True)
    ds.absorbance_corrected.values[:] += 0.0002 * ds.time.values[None, :, None]
    result = fit_dataset(
        ds,
        replace(
            FAST, fit_drift=True, extinction_coefficient_M_cm=6000, signal_direction="decreasing"
        ),
    )
    assert_recovery(result, tolerance=0.02)
    assert result.parameter_rows[0]["drift_AU_per_s"] == pytest.approx(0.0002, abs=1e-6)
    evidence = next(iter(result.run["group_reports"].values()))["identifiability"]
    assert "drift_AU_per_s" in evidence["sensitivity_parameters"]


def test_partial_success_and_no_valid_fit(example):
    ds = example.copy(deep=True)
    ds.absorbance_corrected.values[:, :, 12:24] = 1
    result = fit_dataset(ds, replace(FAST, merge_plates=True))
    assert result.exit_code == 3
    assert {r["status"] for r in result.parameter_rows} == {"success", "ambiguous_direction"}
    ds = example.isel(plate_read=[0], well_384=slice(0, 12)).copy(deep=True)
    ds.expected_conc_um_NORM1.values[0, 0] = np.nan
    result = fit_dataset(ds, FAST)
    assert result.exit_code == 3
    assert result.parameter_rows[0]["status"] == "partial_success"
    ds.expected_conc_um_NORM1.values[:] = 0
    assert fit_dataset(ds, FAST).exit_code == 4


def test_raw_weighting_baseline_and_fixed_kinetic_parameters(single):
    result = fit_dataset(
        single,
        replace(
            FAST,
            weighting="raw_mse",
            fixed_parameters={"Km_mM": 0.3},
            extinction_coefficient_M_cm=6000,
        ),
    )
    row = assert_recovery(result)
    assert row["Km_fixed"] and row["Km_mM"] == 0.3
    assert all(r["scale_AU"] == 1 for r in result.trace_rows)
    for trace in result.trace_rows:
        assert trace["baseline_AU"] == pytest.approx(20, abs=1e-4)
    expected = np.mean([t["rmse_AU"] ** 2 for t in result.trace_rows])
    assert row["loss"] == pytest.approx(expected, abs=1e-15)


def test_no_finite_multistart_group_remains_visible(single, monkeypatch):
    import progress_kinetics.api as api

    monkeypatch.setattr(api, "multistart", lambda *a: (None, [{"status": "nonfinite_failure"}]))
    result = api.fit_dataset(single, FAST)
    assert result.exit_code == 4
    assert result.parameter_rows[0]["status"] == "optimization_failed"
    assert all(t["status"] == "optimization_failed" for t in result.trace_rows)
    assert not result.predictions.used_for_fit.any()
