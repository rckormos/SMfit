"""Sensitivity evidence and optional reoptimized optical profiles, not confidence intervals."""

from __future__ import annotations

import numpy as np
import torch

from .optimize import optimize


def identification(model, batch, attempts):
    c = model.config
    v = model.physical()
    loss, prediction, residual, baseline, s, product, alpha = model.evaluate(batch)
    mask = batch["mask"].bool()
    reasons, columns = [], []
    positive_names = ("Km_mM", "kcat_per_s", "optical")
    eps = 1e-4
    for name in model.active_names:
        plus, minus = dict(v), dict(v)
        if name in positive_names:
            plus[name], minus[name] = v[name] * np.exp(eps), v[name] * np.exp(-eps)
        else:
            span = c.bounds[name][1] - c.bounds[name][0]
            plus[name], minus[name] = v[name] + eps * span, v[name] - eps * span
        with torch.no_grad():
            p1, p0 = model.evaluate(batch, plus)[1], model.evaluate(batch, minus)[1]
            jac = (
                (p1 - p0)
                / (2 * eps)
                / batch["scale"]
                / torch.sqrt(batch["mask"].sum(1, keepdim=True))
            )
            columns.append(jac[mask].cpu().numpy())
    if columns:
        jacobian = np.column_stack(columns)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        norms = np.linalg.norm(jacobian, axis=0)
        normalized = jacobian / np.maximum(norms, 1e-30)
        sensitivity_cosines = (normalized.T @ normalized).tolist()
        condition = float(singular[0] / max(singular[-1], 1e-30))
        rank = int(np.sum(singular > singular[0] / c.sensitivity_condition_limit))
        if rank < len(columns) or condition > c.sensitivity_condition_limit:
            reasons.append("ill_conditioned_profiled_parameter_sensitivity")
    else:
        singular, condition, rank = np.array([]), 1.0, 0
        sensitivity_cosines = []
    with torch.no_grad():
        depletion = (product / batch["s0"]).masked_fill(~mask, 0).max(1).values.cpu().numpy()
    substrates = batch["s0"].detach().cpu().numpy().ravel()
    if np.median(depletion) < 0.1:
        reasons.append("low_depletion_optical_kcat_confounding")
    if np.max(substrates) / np.min(substrates) < 3:
        reasons.append("narrow_substrate_range_Km_recovery_uncertain")
    hits = []
    for name in model.active_names:
        lo, hi = c.bounds[name]
        value = float(v[name].detach())
        fraction = (
            (np.log(value / lo) / np.log(hi / lo))
            if name in positive_names
            else (value - lo) / (hi - lo)
        )
        if min(fraction, 1 - fraction) < 0.005:
            hits.append(name)
    if hits:
        reasons.append("parameter_bound_hit")
    close = [
        a
        for a in attempts
        if "loss" in a and a["loss"] <= max(float(loss.detach()) * 1.1, float(loss.detach()) + 1e-7)
    ]
    spreads = {}
    for name in positive_names:
        vals = [a["parameters"][name] for a in close]
        spreads[name] = float(max(vals) / min(vals)) if vals else None
    if any(ratio is not None and ratio > 2 for ratio in spreads.values()):
        reasons.append("competing_multistart_parameters_with_similar_loss")
    profiles = []
    if c.identifiability == "profile" and "optical" in model.active_names:
        initial = {key: float(value.detach()) for key, value in v.items()}
        for factor in c.profile_factors:
            optical = initial["optical"] * factor
            lo, hi = c.bounds["optical"]
            if not lo <= optical <= hi:
                profiles.append({"optical": optical, "status": "outside_bounds"})
                continue
            fitted, report = optimize(c, batch, initial, fixed_optical=optical)
            profiles.append(
                {
                    "optical": optical,
                    "factor": factor,
                    **{k: val for k, val in report.items() if k != "history"},
                }
            )
        if any(
            row.get("factor", 1) != 1 and row.get("loss", np.inf) <= float(loss.detach()) + 1e-5
            for row in profiles
        ):
            reasons.append("flat_reoptimized_optical_profile")
    res = residual.detach().cpu().numpy()
    autocorrelations = []
    for row, m, times in zip(res, mask.cpu().numpy(), batch["time"].cpu().numpy()):
        vals = row[m][np.argsort(times[m], kind="stable")]
        autocorrelations.append(
            float(np.corrcoef(vals[:-1], vals[1:])[0, 1])
            if len(vals) > 3 and np.std(vals) > 1e-12
            else None
        )
    return {
        "weak_identification": bool(reasons),
        "reasons": reasons,
        "sensitivity_parameters": model.active_names,
        "sensitivity_singular_values": singular.tolist(),
        "sensitivity_rank": rank,
        "sensitivity_column_cosines": sensitivity_cosines,
        "sensitivity_condition": condition,
        "bound_hits": hits,
        "depletion_fraction_per_trace": depletion.tolist(),
        "substrate_range_mM": [float(substrates.min()), float(substrates.max())],
        "enzyme_range_uM": [float(batch["enzyme"].min()), float(batch["enzyme"].max())],
        "near_best_parameter_ratios": spreads,
        "optical_profile": profiles,
        "residual_lag1_correlation_per_trace": autocorrelations,
        "intervals": "not_computed_profiles_are_not_confidence_intervals",
    }
