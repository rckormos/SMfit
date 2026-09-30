from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from progress_kinetics import FitConfig, SchemaError, validate_dataset
from progress_kinetics.data import prepare_dataset


def test_named_dimensions_shuffle_scalar_broadcast_and_nonmutation(single):
    before = single.copy(deep=True)
    single = single.isel(well_384=[11, 0, 3, 2, 1, 7, 6, 5, 4, 10, 9, 8])
    single["extra_metadata"] = "preserved"
    data = prepare_dataset(single, FitConfig())
    assert data.traces[0]["well"] == "A12"
    assert data.traces[0]["S0_mM"] == 3
    assert data.signal[0, 0] == single.absorbance_corrected.isel(plate_read=0, well_384=0, time=0)
    assert len(data.groups) == 1
    xr.testing.assert_identical(
        before, single.sel(well_384=before.well_384).drop_vars("extra_metadata")
    )
    ds = single.copy()
    ds["expected_conc_um_NORM1"] = xr.DataArray(0.4, attrs={"units": "uM"})
    assert all(r["enzyme_uM"] == 0.4 for r in prepare_dataset(ds, FitConfig()).traces)


def test_custom_dimensions_and_unit_overrides(single):
    ds = single.rename(
        {
            "plate_read": "plate",
            "well_384": "trace",
            "time": "sample",
            "cofactor_concentration_mM": "S0",
            "expected_conc_um_NORM1": "E",
        }
    )
    ds.S0.attrs.clear()
    c = FitConfig(
        plate_dim="plate",
        well_dim="trace",
        time_dim="sample",
        time_var="sample",
        substrate_var="S0",
        enzyme_var="E",
        units_overrides={"S0": "mM"},
    )
    assert prepare_dataset(ds, c).traces[0]["S0_mM"] == 0.05
    with pytest.raises(SchemaError, match="missing units"):
        prepare_dataset(ds, replace(c, units_overrides={}))


@pytest.mark.parametrize(
    "variable",
    ["time", "cofactor_concentration_mM", "expected_conc_um_NORM1", "absorbance_corrected"],
)
def test_unknown_units_require_explicit_override(single, variable):
    ds = single.copy(deep=True)
    original = ds[variable].attrs["units"]
    ds[variable].attrs["units"] = "unknown"
    with pytest.raises(SchemaError, match="units"):
        prepare_dataset(ds, FitConfig())
    assert prepare_dataset(ds, FitConfig(units_overrides={variable: original})).used.any()


def test_time_conversion_missing_units_and_plate_time(single):
    ds = single.assign_coords(time=single.time / 60)
    ds.time.attrs["units"] = "min"
    data = prepare_dataset(ds, FitConfig())
    np.testing.assert_allclose(data.time[0], single.time)
    ds.time.attrs.clear()
    assert "missing_time_units_assumed_seconds" in prepare_dataset(ds, FitConfig()).warnings
    ds["elapsed"] = (("plate_read", "time"), np.array([single.time.values + 123]))
    ds.elapsed.attrs["units"] = "s"
    np.testing.assert_allclose(
        prepare_dataset(ds, FitConfig(time_var="elapsed")).time[0], single.time
    )


@pytest.mark.parametrize("dim", ["plate_read", "well_384", "time"])
def test_duplicate_coordinate_rejected(single, dim):
    ds = single
    if dim == "plate_read":
        ds = xr.concat([ds, ds], dim="plate_read", data_vars="all")
    else:
        vals = ds[dim].values.copy()
        vals[1] = vals[0]
        ds = ds.assign_coords({dim: vals})
    with pytest.raises(SchemaError, match="Duplicated"):
        prepare_dataset(ds, FitConfig())


def test_missing_coordinates_and_incompatible_dimensions(single):
    with pytest.raises(SchemaError, match="Missing labeled"):
        prepare_dataset(single.drop_vars("well_384"), FitConfig())
    ds = single.copy()
    ds["expected_conc_um_NORM1"] = ("time", np.ones(ds.sizes["time"]))
    with pytest.raises(SchemaError, match="metadata dimensions"):
        prepare_dataset(ds, FitConfig())
    with pytest.raises(SchemaError, match="exactly"):
        prepare_dataset(
            single.drop_vars("absorbance_corrected").assign(
                absorbance_corrected=single.absorbance_corrected.isel(plate_read=0)
            ),
            FitConfig(),
        )


def test_nonmonotonic_sort_preserves_original_indices(single):
    ds = single.isel(time=[2, 0, 1] + list(range(3, single.sizes["time"])))
    with pytest.raises(SchemaError, match="Nonmonotonic"):
        prepare_dataset(ds, FitConfig())
    data = prepare_dataset(ds, FitConfig(sort_time=True))
    assert data.time[0, 0] == 40
    assert data.time[0, 1] == 0
    assert any("sorted_time" in w for w in data.warnings)


def test_duplicate_time_mean_policy(single):
    ds = single.isel(time=[0, 0] + list(range(1, single.sizes["time"])))
    ds.absorbance_corrected.values[:, 1, :] += 0.2
    data = prepare_dataset(ds, FitConfig(duplicate_time="mean"))
    assert data.masks["duplicate_aggregated"][:, 1].all()
    assert not data.used[:, 1].any()
    np.testing.assert_allclose(data.signal[:, 0], single.absorbance_corrected.values[0, 0] + 0.1)


def test_invalid_missing_concentrations_controls_and_selection(single):
    ds = single.copy(deep=True)
    ds.expected_conc_um_NORM1.values[0, 0] = np.nan
    ds.cofactor_concentration_mM.values[1, 0] = 0
    data = prepare_dataset(ds, FitConfig())
    assert data.traces[0]["status"] == "invalid_metadata"
    assert data.traces[1]["status"] == "control"
    assert not data.used[:2].any()
    ds.expected_conc_um_NORM1.values[0, 0] = -1
    with pytest.raises(SchemaError, match="Negative"):
        prepare_dataset(ds, FitConfig())
    selected = prepare_dataset(single, FitConfig(selections={"PLASMIDID": ["other"]}))
    assert all(r["status"] == "selected_out" for r in selected.traces)


def test_saturation_mask_string_time_and_missing(single):
    ds = single.copy(deep=True)
    ds.saturated.values[0, 10, 0] = True
    ds.absorbance_corrected.values[0, 2, 0] = np.nan
    ds.absorbance_corrected.values[0, 11:, 0] = 123
    data = prepare_dataset(ds, FitConfig())
    assert not data.used[0, 2] and data.used[0, 3]
    assert not data.used[0, 10:].any()
    assert data.traces[0]["trim_time_s"] == 200
    strings = single.copy(deep=True)
    strings["absorbance_corrected"] = single.absorbance_corrected.astype(str)
    strings.absorbance_corrected.values[0, 7, 0] = " ???? "
    data = prepare_dataset(strings, FitConfig())
    assert not data.used[0, 7:].any() and data.used[0, 6]
    timed = single.copy()
    timed["saturation_time_s"] = ("well_384", [40] + [np.nan] * 11)
    data = prepare_dataset(timed, FitConfig())
    assert data.traces[0]["status"] == "insufficient_data"
    assert data.masks["saturation_trimmed"][0, 2:].all()


def test_all_saturated_and_opt_in_missing(single):
    ds = single.copy(deep=True)
    ds.saturated.values[:] = True
    assert not prepare_dataset(ds, FitConfig()).used.any()
    ds = single.copy(deep=True)
    ds.absorbance_corrected.values[0, 9, 0] = np.inf
    default = prepare_dataset(ds, FitConfig())
    assert default.used[0, 10]
    assert any("do_not_imply_saturation" in w for w in default.warnings)
    assert not prepare_dataset(ds, FitConfig(trim_at_first_missing=True)).used[0, 9:].any()


def test_raw_and_corrected_optical_conventions(single):
    ds = single.copy(deep=True)
    ds.attrs.pop("reference_path_length_cm")
    data = prepare_dataset(ds, FitConfig())
    assert np.all(data.lengths == 1)
    assert "assumed_1cm_reference" in data.warnings
    assert data.conventions["epsilon_is_reference_conditional"]
    data = prepare_dataset(ds, FitConfig(reference_path_length_cm=2))
    assert np.all(data.lengths == 2)
    assert not data.conventions["epsilon_is_reference_conditional"]
    raw = prepare_dataset(ds, FitConfig(signal_mode="raw"))
    np.testing.assert_array_equal(raw.lengths, ds.pathlength.values.ravel())
    with pytest.raises(SchemaError, match="pathlength"):
        prepare_dataset(ds.drop_vars("pathlength"), FitConfig(signal_mode="raw"))
    assert np.all(
        prepare_dataset(ds, FitConfig(signal_mode="raw", fixed_path_length_cm=0.4)).lengths == 0.4
    )
    assert (
        prepare_dataset(
            ds.drop_vars("pathlength"), FitConfig(signal_mode="raw", effective_alpha_only=True)
        ).conventions["optical_parameter"]
        == "alpha"
    )


def test_grouping_multiple_enzymes_conditions_and_plate_identity(example):
    data = prepare_dataset(example, FitConfig())
    assert len(data.groups) == 4
    assert len({(r["plate"], r["well"]) for r in data.traces}) == len(data.traces)
    assert len(prepare_dataset(example, FitConfig(merge_plates=True)).groups) == 2
    ds = example.copy()
    ds["mastermix_id"] = ("well_384", ["buffer_a"] * 6 + ["buffer_b"] * 20)
    assert len(prepare_dataset(ds, FitConfig()).groups) == 6
    assert (
        len(prepare_dataset(ds, FitConfig(group_by=("cofactor_name",), merge_plates=True)).groups)
        == 2
    )


def test_explicit_group_preference_and_homogeneity(single):
    ds = single.copy()
    ds["fit_group"] = "explicit"
    data = prepare_dataset(ds, FitConfig())
    assert "fit_group" in next(iter(data.groups))
    ds["PLASMIDID"] = ("well_384", ["one"] * 6 + ["two"] * 6)
    with pytest.raises(SchemaError, match="mixes enzyme"):
        prepare_dataset(ds, FitConfig())
    no_identity = single.drop_vars("PLASMIDID")
    with pytest.raises(SchemaError, match="identity"):
        prepare_dataset(no_identity, FitConfig())
    assert prepare_dataset(no_identity, FitConfig(single_enzyme=True)).used.all()
    no_identity["fit_group"] = "asserted_by_group"
    assert prepare_dataset(no_identity, FitConfig()).used.all()


def test_missing_condition_and_role_mapping(single):
    ds = single.copy()
    ds["cofactor_name"] = ("well_384", ["valid"] * 11 + [""])
    assert prepare_dataset(ds, FitConfig()).traces[-1]["status"] == "invalid_metadata"
    ds["sampletype"] = ("well_384", ["blank"] + ["bad"] + ["sample"] * 10)
    data = prepare_dataset(ds, FitConfig(sample_roles={"blank": "control", "bad": "exclude"}))
    assert data.traces[0]["status"] == "control"
    assert data.traces[1]["status"] == "selected_out"


def test_reaction_age_and_window_do_not_rezero(single):
    ds = single.copy(deep=True)
    ds.absorbance_corrected.values[0, :5, 0] = np.nan
    data = prepare_dataset(
        ds, FitConfig(reaction_age_at_first_read_s={"plate_1": 25}, time_window_s=(60, 700))
    )
    assert data.reaction_time[0, 5] == 125
    assert data.reaction_time[1, 3] == 85
    assert not data.used[1, :3].any()
    with pytest.raises(SchemaError, match="mapping missing"):
        prepare_dataset(ds, FitConfig(reaction_age_at_first_read_s={}))


def test_inspection_temperature_and_grouping_warnings(single):
    ds = single.isel(well_384=[0, 1, 2])
    ds["temperature_C"] = ("time", np.linspace(20, 25, ds.sizes["time"]))
    report = validate_dataset(ds, FitConfig())
    assert any("temperature_varies" in w for w in report["warnings"])
    assert "direction" in report["groups"][0]


def test_concentration_units_ms_and_alignment_missing(single):
    ds = single.copy(deep=True)
    ds.cofactor_concentration_mM.values[:] /= 1000
    ds.cofactor_concentration_mM.attrs["units"] = "M"
    ds.expected_conc_um_NORM1.values[:] /= 1000
    ds.expected_conc_um_NORM1.attrs["units"] = "mM"
    ds = ds.assign_coords(time=ds.time.values * 1000)
    ds.time.attrs["units"] = "ms"
    data = prepare_dataset(ds, FitConfig())
    np.testing.assert_allclose(
        [r["S0_mM"] for r in data.traces], single.cofactor_concentration_mM.values[:, 0]
    )
    np.testing.assert_allclose(
        [r["enzyme_uM"] for r in data.traces], single.expected_conc_um_NORM1.values[0]
    )
    np.testing.assert_allclose(data.time[0], single.time)
    # xarray aligns on assignment; uncovered wells become explicit missing metadata.
    ds = single.copy(deep=True)
    ds["expected_conc_um_NORM1"] = ds.expected_conc_um_NORM1.isel(well_384=slice(1, None))
    assert prepare_dataset(ds, FitConfig()).traces[0]["status"] == "invalid_metadata"


def test_explicit_group_condition_homogeneity_and_deterministic_objects(single):
    ds = single.copy()
    ds["fit_group"] = "one"
    ds["mastermix_id"] = ("well_384", ["one"] * 6 + ["two"] * 6)
    with pytest.raises(SchemaError, match="mixes enzyme/condition mastermix_id"):
        prepare_dataset(ds, FitConfig())
    # Explicit grouping may omit mastermix when it encodes dilution, not condition.
    assert len(prepare_dataset(ds, FitConfig(group_by=("cofactor_name",))).groups) == 1
    from progress_kinetics.data import stable

    assert stable({"a": 1, "b": 2}) == stable({"b": 2, "a": 1})
