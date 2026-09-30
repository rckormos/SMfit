"""Public API. File I/O and plotting are deliberately outside the fit."""

from __future__ import annotations

import json
import platform

import numpy as np
import torch
import xarray as xr

from .config import FitConfig
from .data import prepare_dataset
from .diagnostics import identification
from .direction import detect_direction
from .model import make_batch
from .optimize import multistart
from .results import FitResults, clean_json, versions


def validate_dataset(dataset: xr.Dataset, config: FitConfig | None = None):
    """Inspect schema, selection, grouping, direction evidence and conventions, without fitting."""
    data = prepare_dataset(dataset, config or FitConfig())
    summary = data.summary()
    for row in summary["groups"]:
        indices = [i for i in data.groups[row["group_id"]] if data.traces[i]["status"] == "ready"]
        if indices:
            row["direction"] = detect_direction(data, indices)[1]
    return summary


def fit_dataset(dataset: xr.Dataset, config: FitConfig | None = None) -> FitResults:
    c = config or FitConfig()
    data = prepare_dataset(dataset, c)
    c = data.config
    shape = data.signal.shape
    arrays = {
        name: np.full(shape, np.nan)
        for name in ("prediction_AU", "residual_AU", "substrate_mM", "product_mM")
    }
    used = np.zeros(shape, dtype=bool)
    rows, reports = [], {}
    for group, all_indices in data.groups.items():
        indices = [i for i in all_indices if data.traces[i]["status"] == "ready"]
        row = {
            "group_id": group,
            "status": "no_kinetic_traces",
            "reason": "no usable kinetic traces",
            "Km_mM": np.nan,
            "kcat_per_s": np.nan,
            "epsilon_per_M_cm": np.nan,
            "alpha_AU_per_mM": np.nan,
            "direction": "ambiguous",
            "loss": np.nan,
            "converged": False,
            "weak_identification": True,
            "Km_fixed": "Km_mM" in c.fixed_parameters,
            "kcat_fixed": "kcat_per_s" in c.fixed_parameters,
            "optical_fixed": c.extinction_coefficient_M_cm is not None
            or c.fix_alpha_AU_per_mM is not None,
            "epsilon_is_reference_conditional": data.conventions[
                "epsilon_is_reference_conditional"
            ],
            "reference_source": data.conventions["reference_source"],
            "reporter_stoichiometry": c.reporter_stoichiometry,
            "enzyme_basis": c.enzyme_basis,
            "kinetic_trace_count": len(indices),
            "total_trace_count": len(all_indices),
        }
        reports[group] = {}
        if not indices:
            rows.append(row)
            continue
        direction, direction_report = detect_direction(data, indices)
        reports[group]["direction"] = direction_report
        row["direction"] = direction_report["direction"]
        if direction is None:
            row.update(
                status="ambiguous_direction", reason="insufficient or conflicting sign votes"
            )
            for i in indices:
                data.traces[i].update(status="ambiguous_direction", reason=row["reason"])
            rows.append(row)
            continue
        batch = make_batch(data, indices, direction)
        model, attempts = multistart(c, batch)
        reports[group]["multistart"] = attempts
        if model is None:
            row.update(status="optimization_failed", reason="no finite evaluated state")
            for i in indices:
                data.traces[i].update(status="optimization_failed", reason=row["reason"])
            rows.append(row)
            continue
        evidence = identification(model, batch, attempts)
        reports[group]["identifiability"] = evidence
        best = min((a for a in attempts if "loss" in a), key=lambda a: a["loss"])
        values = {key: float(v.detach()) for key, v in model.physical().items()}
        with torch.no_grad():
            loss, pred, residual, baseline, s, product, alpha = model.evaluate(batch)
        invalid = any(
            data.traces[i]["status"] in ("invalid_metadata", "insufficient_data")
            for i in all_indices
        )
        row.update(
            status="partial_success" if invalid else "success",
            reason="some requested traces unusable" if invalid else "",
            Km_mM=values["Km_mM"],
            kcat_per_s=values["kcat_per_s"],
            epsilon_per_M_cm=np.nan if model.alpha_mode else values["optical"],
            alpha_AU_per_mM=float(alpha.mean()),
            alpha_min_AU_per_mM=float(alpha.min()),
            alpha_max_AU_per_mM=float(alpha.max()),
            lag_s=values["lag_s"],
            drift_AU_per_s=values["drift_AU_per_s"],
            loss=float(loss),
            converged=best["converged"],
            termination=best["termination"],
            weak_identification=evidence["weak_identification"],
            identifiability_reasons=evidence["reasons"],
            bound_hits=evidence["bound_hits"],
            sensitivity_condition=evidence["sensitivity_condition"],
        )
        for name, value in (("prediction_AU", pred), ("substrate_mM", s), ("product_mM", product)):
            arrays[name][indices] = value.cpu().numpy()
        arrays["residual_AU"][indices] = np.where(
            data.used[indices], data.signal[indices] - pred.cpu().numpy(), np.nan
        )
        used[indices] = data.used[indices]
        for j, i in enumerate(indices):
            mask = data.used[i]
            raw_residual = residual[j].cpu().numpy()[mask]
            noise_report = direction_report["trace_votes"][j]
            data.traces[i].update(
                status="fitted",
                baseline_AU=float(baseline[j, 0]),
                Vmax_mM_per_s=values["kcat_per_s"] * data.traces[i]["enzyme_uM"] / 1000,
                alpha_AU_per_mM=float(alpha[j, 0]),
                rmse_AU=float(np.sqrt(np.mean(raw_residual**2))),
                mean_residual_AU=float(raw_residual.mean()),
                dynamic_range_AU=float(np.ptp(data.signal[i, mask])),
                noise_AU=noise_report["noise_AU"],
                scale_AU=float(batch["scale"][j, 0]),
                trace_weight=float(1 / len(indices)),
                point_weight=float(1 / mask.sum()),
                depletion_fraction=evidence["depletion_fraction_per_trace"][j],
                residual_lag1_correlation=evidence["residual_lag1_correlation_per_trace"][j],
            )
        rows.append(row)
    p, w, n = (dataset.sizes[d] for d in (c.plate_dim, c.well_dim, c.time_dim))
    dims = (c.plate_dim, c.well_dim, c.time_dim)
    predictions = dataset.copy(deep=True)

    def add(name, values, units=None):
        predictions[name] = (dims, values.reshape(p, w, n))
        predictions[name] = predictions[name].transpose(c.plate_dim, c.time_dim, c.well_dim)
        if units:
            predictions[name].attrs["units"] = units

    for name, values in arrays.items():
        add(name, values, "mM" if name.endswith("mM") else "AU")
    add("observed_AU", data.signal, "AU")
    add("acquisition_time_s", data.time, "s")
    add("reaction_time_s", data.reaction_time, "s")
    add("used_for_fit", used)
    for name, values in data.masks.items():
        add(name, values)
    predictions = predictions.assign_coords(original_sample_index=(c.time_dim, np.arange(n)))
    parameter_data = {}
    for key in list(dict.fromkeys(key for row in rows for key in row)):
        if key == "group_id":
            continue
        vals = [row.get(key, np.nan) for row in rows]
        if all(isinstance(v, (float, int, bool, np.number)) for v in vals):
            parameter_data[key] = ("group", np.array(vals))
        else:
            parameter_data[key] = (
                "group",
                np.array(
                    [
                        json.dumps(clean_json(v)) if isinstance(v, (list, dict)) else str(v)
                        for v in vals
                    ]
                ),
            )
    parameters = xr.Dataset(parameter_data, coords={"group": [row["group_id"] for row in rows]})
    diagnostics = xr.Dataset(coords={d: dataset[d] for d in dims[:2]})
    for key in list(dict.fromkeys(key for row in data.traces for key in row)):
        vals = [row.get(key) for row in data.traces]
        if all(v is None or isinstance(v, (float, int, np.number)) for v in vals):
            values = np.array([np.nan if v is None else v for v in vals], dtype=float)
        else:
            values = np.array(["" if v is None else str(v) for v in vals], dtype=str)
        diagnostics[f"trace_{key}"] = (dims[:2], values.reshape(p, w))
    run = {
        "config": c.to_dict(),
        "versions": versions(),
        "python": platform.python_version(),
        "seed": c.seed,
        "conventions": data.conventions,
        "warnings": data.warnings,
        "group_statuses": rows,
        "group_reports": reports,
        "input_attrs": dict(dataset.attrs),
        "schema_version": 1,
        "status": "success"
        if all(r["status"] == "success" for r in rows)
        else "partial_success"
        if any(r["status"] in ("success", "partial_success") for r in rows)
        else "no_valid_fit",
    }
    return FitResults(parameters, predictions, diagnostics, rows, data.traces, clean_json(run))
