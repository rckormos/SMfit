"""Named-dimension adaptation and auditable preprocessing without input mutation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

from .config import FitConfig


class SchemaError(ValueError):
    """Input schema cannot be interpreted safely."""


def stable(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "<missing>"
    if isinstance(value, bytes):
        value = value.decode("utf8")
    text = str(value).strip()
    return text if text else "<missing>"


@dataclass
class PreparedData:
    dataset: xr.Dataset
    config: FitConfig
    signal: np.ndarray  # trace, original sample
    time: np.ndarray  # acquisition seconds, same shape
    reaction_time: np.ndarray
    used: np.ndarray
    masks: dict[str, np.ndarray]
    traces: list[dict]
    groups: dict[str, list[int]]
    warnings: list[str]
    conventions: dict
    lengths: np.ndarray

    def summary(self):
        summaries = []
        for group, indices in self.groups.items():
            rows = [self.traces[i] for i in indices]
            usable = [r for r in rows if r["status"] == "ready"]
            summaries.append(
                {
                    "group_id": group,
                    "trace_count": len(rows),
                    "kinetic_trace_count": len(usable),
                    "substrate_range_mM": _range([r["S0_mM"] for r in usable]),
                    "enzyme_range_uM": _range([r["enzyme_uM"] for r in usable]),
                }
            )
        return {
            "groups": summaries,
            "traces": self.traces,
            "warnings": self.warnings,
            "conventions": self.conventions,
        }


def _range(values):
    return [float(min(values)), float(max(values))] if values else []


def prepare_dataset(dataset: xr.Dataset, config: FitConfig) -> PreparedData:
    c = config
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset")
    if c.group_var is None and "fit_group" in dataset:
        c = replace(c, group_var="fit_group")
    dims = (c.plate_dim, c.well_dim, c.time_dim)
    if len(set(dims)) != 3:
        raise SchemaError("Plate, well and time dimensions must be distinct")
    for dim in dims:
        if dim not in dataset.dims or dim not in dataset.coords:
            raise SchemaError(f"Missing labeled dimension/coordinate {dim!r}")
        coord = dataset[dim]
        if coord.dims != (dim,) or coord.size == 0:
            raise SchemaError(f"{dim}: expected a nonempty one-dimensional coordinate")
        labels = [stable(v) for v in coord.values]
        if "<missing>" in labels:
            raise SchemaError(f"Missing coordinate in {dim}")
        if len(set(labels)) != len(labels) and not (
            dim == c.time_dim and c.duplicate_time == "mean"
        ):
            raise SchemaError(f"Duplicated coordinate in {dim}")
    warnings = []
    p, w, n = (dataset.sizes[d] for d in dims)
    template = xr.DataArray(
        np.zeros((p, w)), dims=dims[:2], coords={d: dataset[d] for d in dims[:2]}
    )

    def metadata(name, *, default=None, required=False):
        if name not in dataset:
            if required:
                raise SchemaError(f"Missing required variable {name!r}")
            return np.full(p * w, default, dtype=object)
        da = dataset[name]
        if not set(da.dims) <= set(dims[:2]):
            raise SchemaError(f"{name}: metadata dimensions must be scalar, plate or well")
        try:
            da, _ = xr.align(da, template, join="exact")
            return da.broadcast_like(template).transpose(*dims[:2]).values.reshape(-1).copy()
        except (ValueError, KeyError) as exc:
            raise SchemaError(f"Alignment error for {name}: {exc}") from exc

    def point_data(name):
        if name not in dataset:
            raise SchemaError(f"Missing variable {name!r}")
        da = dataset[name]
        if set(da.dims) != set(dims):
            raise SchemaError(f"{name} must have exactly plate, time and well dimensions")
        return da.transpose(*dims).values.reshape(p * w, n).copy()

    def unit(name, expected, accepted, implied=False):
        supplied = c.units_overrides.get(name, dataset[name].attrs.get("units"))
        if supplied is None:
            if implied:
                warnings.append(f"{name}: assumed_{expected}_units_from_default_mapping")
                return 1.0
            raise SchemaError(f"{name}: missing units; supply units_overrides")
        key = str(supplied).strip().lower().replace("μ", "u").replace("µ", "u")
        if key not in accepted:
            raise SchemaError(
                f"{name}: unknown/incompatible units {supplied!r}; expected {expected}"
            )
        return accepted[key]

    raw_signal = point_data(c.signal_var)
    unit(
        c.signal_var,
        "AU",
        {"au": 1, "absorbance": 1, "1": 1},
        implied=c.signal_var == "absorbance_corrected",
    )
    signal = np.full((p * w, n), np.nan)
    markers = np.zeros_like(signal, dtype=bool)
    for index, value in np.ndenumerate(raw_signal):
        if isinstance(value, (str, bytes)):
            text = stable(value)
            if re.fullmatch(r"\?{2,}", text):
                markers[index] = True
                continue
        try:
            signal[index] = float(value)
        except (ValueError, TypeError):
            if stable(value) != "<missing>":
                raise SchemaError(f"Unrecognized signal value at trace/sample {index}: {value!r}")
    if c.saturation_var and c.saturation_var in dataset:
        sat = point_data(c.saturation_var)
        if not np.isin(sat, [False, True, 0, 1]).all():
            raise SchemaError("Saturation mask must contain only booleans/0/1")
        markers |= sat.astype(bool)
    if c.time_var not in dataset:
        raise SchemaError(f"Missing time variable {c.time_var}")
    time_da = dataset[c.time_var]
    if c.time_dim not in time_da.dims or not set(time_da.dims) <= {c.plate_dim, c.time_dim}:
        raise SchemaError("Time must be time-only or plate/time; use an explicit mapping")
    time_units = c.units_overrides.get(c.time_var, time_da.attrs.get("units"))
    if time_units is None:
        warnings.append("missing_time_units_assumed_seconds")
        time_factor = 1.0
    else:
        time_factor = unit(
            c.time_var,
            "seconds",
            {
                "s": 1,
                "sec": 1,
                "seconds": 1,
                "second": 1,
                "ms": 0.001,
                "milliseconds": 0.001,
                "min": 60,
                "minutes": 60,
                "minute": 60,
            },
        )
    time_template = xr.DataArray(
        np.zeros((p, n)),
        dims=(c.plate_dim, c.time_dim),
        coords={d: dataset[d] for d in (c.plate_dim, c.time_dim)},
    )
    try:
        time_da, _ = xr.align(time_da, time_template, join="exact")
        plate_time = (
            time_da.broadcast_like(time_template)
            .transpose(c.plate_dim, c.time_dim)
            .values.astype(float)
            * time_factor
        )
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"Time alignment/conversion error: {exc}") from exc
    if not np.isfinite(plate_time).all():
        raise SchemaError("Time must be finite numeric seconds")
    for k, times in enumerate(plate_time):
        if np.any(np.diff(times) < 0):
            if not c.sort_time:
                raise SchemaError(f"Nonmonotonic time on plate {k}; select sort_time explicitly")
            warnings.append(f"plate {k}: sorted_time_for_preprocessing_original_indices_preserved")
        if len(np.unique(times)) < 3:
            raise SchemaError(f"Plate {k} has fewer than three distinct time points")
        if len(np.unique(times)) != n and c.duplicate_time == "reject":
            raise SchemaError(f"Plate {k}: duplicate time values")
    plate_time -= np.min(plate_time, axis=1, keepdims=True)
    time = np.repeat(plate_time, w, axis=0)
    plate_labels = [stable(v) for v in dataset[c.plate_dim].values]
    well_labels = [stable(v) for v in dataset[c.well_dim].values]
    ages = c.reaction_age_at_first_read_s
    if isinstance(ages, dict):
        missing = set(plate_labels) - set(ages)
        if missing:
            raise SchemaError(f"Reaction age mapping missing plates: {sorted(missing)}")
        age_values = [ages[label] for label in plate_labels]
    else:
        age_values = [ages] * p
    reaction_time = time + np.repeat(age_values, w)[:, None]
    s0 = metadata(c.substrate_var, required=True)
    enzymes = metadata(c.enzyme_var, required=True)
    s_factor = unit(
        c.substrate_var,
        "mM",
        {"mm": 1, "mmol/l": 1, "m": 1000, "um": 0.001},
        implied=c.substrate_var == "cofactor_concentration_mM",
    )
    e_factor = unit(
        c.enzyme_var,
        "uM",
        {"um": 1, "umol/l": 1, "mm": 1000, "m": 1e6},
        implied=c.enzyme_var == "expected_conc_um_NORM1",
    )
    try:
        s0, enzymes = s0.astype(float) * s_factor, enzymes.astype(float) * e_factor
    except (TypeError, ValueError) as exc:
        raise SchemaError("Concentrations must be numeric; use NaN for missing metadata") from exc
    if np.any(s0 < 0) or np.any(enzymes < 0):
        raise SchemaError("Negative substrate/enzyme concentrations are errors, not controls")
    ids = metadata(c.identity_var)
    if c.identity_var not in dataset and not (c.group_var or c.single_enzyme):
        raise SchemaError("Missing enzyme identity: supply group_var or single_enzyme=True")
    condition_names = [
        v
        for v in ("cofactor_name", "mastermix_id", "wavelength_nm", "NaPO3_concentration_mM")
        if v in dataset
    ]
    if c.group_by is not None:
        group_keys = list(c.group_by)
        for name in group_keys:
            metadata(name, required=True)
        declared_conditions = [v for v in group_keys if v != c.identity_var]
    else:
        group_keys = [c.identity_var] + condition_names
        declared_conditions = condition_names
    if c.identity_var not in group_keys:
        group_keys.insert(0, c.identity_var)
    all_meta = {key: metadata(key) for key in set(group_keys + declared_conditions)}
    group_labels = metadata(c.group_var, required=True) if c.group_var else None
    roles = metadata(c.sample_role_var)
    selected = np.ones(p * w, dtype=bool)
    for name, values in c.selections.items():
        arr = metadata(name, required=True)
        selected &= np.isin([stable(x) for x in arr], [stable(x) for x in values])
    lengths = np.ones(p * w)
    conditional = False
    reference_source = "not_applicable"
    if c.signal_mode == "corrected" and not c.effective_alpha_only:
        reference = c.reference_path_length_cm
        reference_source = "config"
        if reference is None:
            reference = dataset[c.signal_var].attrs.get(
                "reference_path_length_cm", dataset.attrs.get("reference_path_length_cm")
            )
            reference_source = "metadata"
        if reference is None:
            reference = 1.0
            reference_source = "assumed_1cm_reference"
            warnings.append("assumed_1cm_reference")
            conditional = True
        try:
            reference = float(reference)
        except (TypeError, ValueError) as exc:
            raise SchemaError("Reference length must be numeric cm") from exc
        if not np.isfinite(reference) or reference <= 0:
            raise SchemaError("Reference length must be finite positive cm")
        lengths[:] = reference
    elif c.signal_mode == "raw" and not c.effective_alpha_only:
        if c.fixed_path_length_cm is not None:
            lengths[:] = c.fixed_path_length_cm
        else:
            lengths = metadata(c.pathlength_var, required=True).astype(float)
            lengths *= unit(c.pathlength_var, "cm", {"cm": 1, "mm": 0.1, "m": 100})
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
            raise SchemaError(
                "Raw AU requires finite positive pathlengths, or effective_alpha_only"
            )
    masks = {
        name: np.zeros_like(signal, dtype=bool)
        for name in (
            "saturation_trimmed",
            "missing",
            "selected_out",
            "time_window_excluded",
            "control",
            "insufficient_data",
            "duplicate_aggregated",
            "invalid_metadata",
        )
    }
    masks["missing"] = ~np.isfinite(signal)
    sat_times = (
        metadata(c.saturation_time_var)
        if (c.saturation_time_var and c.saturation_time_var in dataset)
        else np.full(p * w, np.nan)
    )
    if c.saturation_time_var and c.saturation_time_var in dataset:
        factor = unit(
            c.saturation_time_var,
            "seconds",
            {"s": 1, "seconds": 1, "ms": 0.001, "min": 60},
            implied=c.saturation_time_var == "saturation_time_s",
        )
        try:
            sat_times = sat_times.astype(float) * factor
        except (ValueError, TypeError) as exc:
            raise SchemaError("Saturation times must be numeric") from exc
    traces, groups = [], {}
    for i in range(p * w):
        plate, well = plate_labels[i // w], well_labels[i % w]
        identity = stable(ids[i])
        fields = {key: stable(all_meta[key][i]) for key in group_keys}
        if identity == "<missing>" and c.single_enzyme:
            fields[c.identity_var] = "asserted_single_enzyme"
        if c.group_var:
            fields = {c.group_var: stable(group_labels[i])}
        if not c.merge_plates:
            fields = {c.plate_dim: plate, **fields}
        group = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        groups.setdefault(group, []).append(i)
        trim_time, trim_cause = None, None
        candidates = []
        if markers[i].any():
            candidates.append((float(time[i, markers[i]].min()), "saturation_marker_or_mask"))
        if np.isfinite(sat_times[i]):
            if sat_times[i] < 0:
                raise SchemaError(f"{plate}/{well}: negative saturation_time_s")
            candidates.append((float(sat_times[i]), "saturation_time"))
        if c.trim_at_first_missing and masks["missing"][i].any():
            candidates.append((float(time[i, masks["missing"][i]].min()), "first_missing_opt_in"))
        if candidates:
            trim_time, trim_cause = min(candidates)
            masks["saturation_trimmed"][i] = time[i] >= trim_time
        role = c.sample_roles.get(stable(roles[i]), "kinetic")
        masks["selected_out"][i] = not selected[i] or role == "exclude"
        masks["control"][i] = s0[i] == 0 or enzymes[i] == 0 or role == "control"
        lo, hi = c.time_window_s
        masks["time_window_excluded"][i] = ((time[i] < lo) if lo is not None else False) | (
            (time[i] > hi) if hi is not None else False
        )
        missing_meta = []
        if not np.isfinite(s0[i]):
            missing_meta.append(c.substrate_var)
        if not np.isfinite(enzymes[i]):
            missing_meta.append(c.enzyme_var)
        if identity == "<missing>" and not (c.group_var or c.single_enzyme):
            missing_meta.append(c.identity_var)
        if c.group_var and stable(group_labels[i]) == "<missing>":
            missing_meta.append(c.group_var)
        if any(v == "<missing>" for key, v in fields.items() if key != c.identity_var):
            missing_meta.append("grouping_metadata")
        masks["invalid_metadata"][i] = bool(missing_meta)
        preliminary = ~np.logical_or.reduce([mask[i] for mask in masks.values()])
        if c.duplicate_time == "mean":
            for t in np.unique(time[i, preliminary]):
                members = np.flatnonzero(preliminary & (time[i] == t))
                if len(members) > 1:
                    signal[i, members[0]] = np.mean(signal[i, members])
                    masks["duplicate_aggregated"][i, members[1:]] = True
                    warnings.append(f"{plate}/{well}: mean_aggregated_duplicate_time={t:g}")
            preliminary &= ~masks["duplicate_aggregated"][i]
        points = int(preliminary.sum())
        status, reason = "ready", ""
        if missing_meta:
            status, reason = "invalid_metadata", "missing/nonfinite: " + ",".join(missing_meta)
        elif masks["selected_out"][i].any():
            status, reason = "selected_out", "explicit_selection_or_sample_role"
        elif masks["control"][i].any():
            status, reason = "control", "zero_concentration_or_explicit_control_role"
        elif points < c.minimum_points:
            status, reason = "insufficient_data", f"{points} usable distinct points"
            masks["insufficient_data"][i] = True
        traces.append(
            {
                "plate": plate,
                "well": well,
                "group_id": group,
                "enzyme_id": identity,
                "S0_mM": float(s0[i]),
                "enzyme_uM": float(enzymes[i]),
                "point_count": points,
                "status": status,
                "reason": reason,
                "trim_time_s": trim_time,
                "trim_cause": trim_cause,
                "sample_role": role,
                "length_cm": float(lengths[i]),
                **{f"n_{key}": int(mask[i].sum()) for key, mask in masks.items()},
            }
        )
    if c.group_var:
        for group, indices in groups.items():
            ready = [i for i in indices if traces[i]["status"] == "ready"]
            for key in [c.identity_var] + declared_conditions:
                vals = ids if key == c.identity_var else all_meta[key]
                known = {stable(vals[i]) for i in ready} - {"<missing>"}
                if len(known) > 1:
                    raise SchemaError(
                        f"Explicit group {group} mixes enzyme/condition {key}: {known}"
                    )
    if not c.group_var and c.group_by is None:
        eligible_groups = [
            inds for inds in groups.values() if any(traces[i]["status"] == "ready" for i in inds)
        ]
        narrow = sum(
            len({s0[i] for i in inds if traces[i]["status"] == "ready"}) < 2
            for inds in eligible_groups
        )
        if eligible_groups and narrow / len(eligible_groups) >= 0.5:
            warnings.append("default_grouping_many_single_substrate_groups_consider_group_by")
    if np.any(masks["missing"]) and not markers.any() and not np.isfinite(sat_times).any():
        warnings.append("missing_values_do_not_imply_saturation_later_finite_readings_retained")
    if "temperature_C" in dataset:
        temp = dataset["temperature_C"].values.astype(float)
        finite = temp[np.isfinite(temp)]
        if finite.size and np.ptp(finite) > 2:
            warnings.append("temperature_varies_more_than_2C_review_stable_assay_assumption")
    if not any(age_values):
        warnings.append("reaction_start_assumed_at_first_acquisition")
    if c.reporter_stoichiometry == 1:
        warnings.append("reporter_stoichiometry_assumed_1")
    conventions = {
        "signal_mode": c.signal_mode,
        "reference_source": reference_source,
        "epsilon_is_reference_conditional": conditional,
        "reporter_stoichiometry": c.reporter_stoichiometry,
        "optical_parameter": "alpha"
        if c.effective_alpha_only or c.fix_alpha_AU_per_mM is not None
        else "epsilon",
        "enzyme_basis": c.enzyme_basis,
        "reaction_age_at_first_read_s": ages,
        "lag_s": c.lag_s,
        "fit_lag": c.fit_lag,
        "fit_drift": c.fit_drift,
        "time_origin": "first_plate_acquisition",
        "saturation_time_basis": "seconds_since_first_plate_acquisition",
        "selected_variables": {
            k: getattr(c, k)
            for k in ("signal_var", "time_var", "substrate_var", "enzyme_var", "identity_var")
        },
    }
    used = ~np.logical_or.reduce(list(masks.values()))
    return PreparedData(
        dataset,
        c,
        signal,
        time,
        reaction_time,
        used,
        masks,
        traces,
        groups,
        list(dict.fromkeys(warnings)),
        conventions,
        lengths,
    )
