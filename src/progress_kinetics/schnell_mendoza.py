"""Differentiable Schnell-Mendoza progress-curve calculations."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _as_tensor_like(value: float | Tensor, reference: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def substrate_concentration(
    time_s: Tensor,
    substrate0_mM: float | Tensor,
    vmax_mM_s: float | Tensor,
    km_mM: float | Tensor,
    *,
    iterations: int = 16,
    concentration_floor_mM: float = 1e-12,
) -> Tensor:
    """Return substrate concentration for irreversible Michaelis-Menten kinetics.

    The Schnell-Mendoza equation is solved in log-concentration coordinates::

        exp(x) + Km*x = S0 + Km*log(S0) - Vmax*t,
        x = log(S).

    Solving for ``x`` avoids the unstable ``Km / S`` term that appears in a
    direct Newton solve when substrate is nearly exhausted. The fixed iteration
    count keeps the operation differentiable with PyTorch autograd.

    Parameters are broadcast using ordinary PyTorch semantics. Concentrations,
    ``vmax_mM_s``, and ``km_mM`` must be positive; time must be nonnegative.
    """

    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if concentration_floor_mM <= 0:
        raise ValueError("concentration_floor_mM must be positive.")
    if not torch.is_floating_point(time_s):
        raise TypeError("time_s must be a floating-point tensor.")

    substrate0 = _as_tensor_like(substrate0_mM, time_s)
    vmax = _as_tensor_like(vmax_mM_s, time_s)
    km = _as_tensor_like(km_mM, time_s)

    if torch.any(time_s < 0):
        raise ValueError("time_s must be nonnegative.")
    if torch.any(substrate0 <= 0):
        raise ValueError("substrate0_mM must be positive.")
    if torch.any(vmax <= 0):
        raise ValueError("vmax_mM_s must be positive.")
    if torch.any(km <= 0):
        raise ValueError("km_mM must be positive.")

    floor = _as_tensor_like(concentration_floor_mM, time_s)
    log_floor = math.log(concentration_floor_mM)
    log_s0 = torch.log(torch.clamp(substrate0, min=floor))
    rhs = substrate0 + km * log_s0 - vmax * time_s

    # A first-order integrated-rate estimate provides a useful starting point
    # in both the saturated and low-substrate regimes.
    x = log_s0 - vmax * time_s / (km + substrate0)
    x = torch.maximum(x, torch.as_tensor(log_floor, dtype=x.dtype, device=x.device))
    x = torch.minimum(x, log_s0)

    for _ in range(iterations):
        s = torch.exp(x)
        residual = s + km * x - rhs
        derivative = s + km
        step = residual / derivative
        # A bounded Newton step avoids rare excursions caused by extreme trial
        # parameters during optimization while preserving useful gradients.
        step = torch.clamp(step, min=-20.0, max=20.0)
        x = torch.maximum(x - step, torch.as_tensor(log_floor, dtype=x.dtype, device=x.device))
        x = torch.minimum(x, log_s0)

    return torch.exp(x)


def product_concentration(
    time_s: Tensor,
    substrate0_mM: float | Tensor,
    vmax_mM_s: float | Tensor,
    km_mM: float | Tensor,
    **solver_options: object,
) -> Tensor:
    """Return product concentration assuming one-to-one substrate conversion."""

    substrate0 = _as_tensor_like(substrate0_mM, time_s)
    substrate = substrate_concentration(
        time_s,
        substrate0,
        vmax_mM_s,
        km_mM,
        **solver_options,
    )
    return torch.clamp(substrate0 - substrate, min=0.0)
