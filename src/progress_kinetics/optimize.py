"""Reproducible Adam -> L-BFGS; history and saved state share an evaluation."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .model import GroupModel


def optimize(config, batch, initial, *, fixed_optical=None):
    model = GroupModel(config, initial, fixed_optical=fixed_optical)
    history = []
    best_loss = math.inf
    best_state = None
    evaluations = 0
    error = None
    params = list(model.parameters())

    def evaluate(stage, backward=True):
        nonlocal best_loss, best_state, evaluations
        model.zero_grad(set_to_none=True)
        loss = model.evaluate(batch)[0]
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite_objective")
        if backward and params:
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("nonfinite_gradient")
        value = float(loss.detach())
        gradient = math.sqrt(
            sum(float(p.grad.square().sum()) for p in params if p.grad is not None)
        )
        values = {key: float(v.detach()) for key, v in model.physical().items()}
        evaluations += 1
        history.append(
            {
                "stage": stage,
                "evaluation": evaluations,
                "loss": value,
                "gradient_norm": gradient,
                **values,
            }
        )
        if value < best_loss:
            best_loss, best_state = value, copy.deepcopy(model.state_dict())
        return loss

    try:
        evaluate("initial")
        if params:
            adam = torch.optim.Adam(params, lr=config.learning_rate)
            for _ in range(config.adam_steps):
                evaluate("adam")
                torch.nn.utils.clip_grad_norm_(params, 10, error_if_nonfinite=True)
                adam.step()
            evaluate("post_adam")
            model.load_state_dict(best_state)
            if config.lbfgs_steps:
                lbfgs = torch.optim.LBFGS(
                    params,
                    max_iter=config.lbfgs_steps,
                    tolerance_grad=config.tolerance_grad,
                    tolerance_change=config.tolerance_change,
                    line_search_fn="strong_wolfe",
                )
                lbfgs.step(lambda: evaluate("lbfgs"))
            evaluate("final_candidate")
    except (FloatingPointError, RuntimeError) as exc:
        error = str(exc)
    if best_state is None:
        return None, {
            "status": "nonfinite_failure",
            "error": error,
            "history": history,
            "initial": initial,
        }
    model.load_state_dict(best_state)
    loss = model.evaluate(batch)[0]
    model.zero_grad(set_to_none=True)
    if params:
        loss.backward()
    grad = math.sqrt(sum(float(p.grad.square().sum()) for p in params if p.grad is not None))
    # A small loss is evidence of a near-exact numerical fit, not model validity.
    converged = not params or grad <= max(config.tolerance_grad * 10, 1e-7)
    values = {key: float(v.detach()) for key, v in model.physical().items()}
    report = {
        "status": "best_finite_restored" if error else "finished",
        "error": error,
        "initial": initial,
        "parameters": values,
        "loss": float(loss.detach()),
        "gradient_norm": grad,
        "converged": converged,
        "evaluations": evaluations,
        "history": history,
        "termination": (
            "nonfinite_or_runtime_failure_restored_best"
            if error
            else "gradient_tolerance"
            if converged
            else "budget_or_change_tolerance"
        ),
    }
    return model, report


def multistart(config, batch):
    rng = np.random.default_rng(config.seed)
    initial = dict(config.starts)
    # Explicit alpha-only fits use AU/mM. The standard start is an epsilon magnitude.
    if config.effective_alpha_only and initial["optical"] == 6000:
        initial["optical"] = 6.0
    attempts, best_model, best_loss = [], None, math.inf
    for start in range(config.multistart):
        candidate = dict(initial)
        if start:
            for name in ("Km_mM", "kcat_per_s", "optical"):
                lo, hi = config.bounds[name]
                candidate[name] = float(
                    np.clip(candidate[name] * np.exp(rng.normal(0, 1.5)), lo * 1.001, hi / 1.001)
                )
        model, report = optimize(config, batch, candidate)
        report["start_index"] = start
        attempts.append(report)
        if model is not None and report["loss"] < best_loss:
            best_model, best_loss = model, report["loss"]
    return best_model, attempts
