"""Generic per-well enzyme model, bounded parameters and profiled baselines."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .schnell_mendoza import substrate_concentration


def substrate(time_s, s0_mM, enzyme_uM, km_mM, kcat_per_s):
    """Broadcast S(t); bypass positive-only solver for exact zero controls."""
    s0 = torch.as_tensor(s0_mM, dtype=time_s.dtype, device=time_s.device)
    e = torch.as_tensor(enzyme_uM, dtype=time_s.dtype, device=time_s.device)
    km = torch.as_tensor(km_mM, dtype=time_s.dtype, device=time_s.device)
    kcat = torch.as_tensor(kcat_per_s, dtype=time_s.dtype, device=time_s.device)
    for name, value in (("time", time_s), ("S0", s0), ("enzyme", e), ("Km", km), ("kcat", kcat)):
        if not torch.isfinite(value).all() or torch.any(value < 0):
            raise ValueError(f"{name} must be finite nonnegative")
    if torch.any(km <= 0) or torch.any(kcat <= 0):
        raise ValueError("Km and kcat must be positive")
    active = (s0 > 0) & (e > 0)
    safe_s0 = torch.where(active, s0, torch.ones_like(s0))
    vmax = kcat * torch.where(active, e, torch.ones_like(e)) / 1000.0
    result = substrate_concentration(time_s, safe_s0, vmax, km, iterations=24)
    return torch.where(active, torch.minimum(result, s0), s0)


def epsilon_to_alpha(epsilon_per_M_cm, length_cm, stoichiometry=1.0):
    """Positive effective extinction change [M^-1 cm^-1] -> AU/mM."""
    return epsilon_per_M_cm * length_cm * stoichiometry / 1000.0


class GroupModel(nn.Module):
    def __init__(self, config, initial, *, fixed_optical=None):
        super().__init__()
        self.config = config
        self.alpha_mode = config.effective_alpha_only or config.fix_alpha_AU_per_mM is not None
        self.fixed = dict(config.fixed_parameters)
        if fixed_optical is not None:
            self.fixed["optical"] = fixed_optical
        elif config.fix_alpha_AU_per_mM is not None:
            self.fixed["optical"] = config.fix_alpha_AU_per_mM
        elif config.extinction_coefficient_M_cm is not None:
            self.fixed["optical"] = config.extinction_coefficient_M_cm
        if not config.fit_lag:
            self.fixed["lag_s"] = config.lag_s
        if not config.fit_drift:
            self.fixed["drift_AU_per_s"] = 0.0
        dtype = getattr(torch, config.dtype)
        self.raw = nn.ParameterDict()
        self.active_names = []
        for name in config.bounds:
            if name in self.fixed:
                continue
            self.active_names.append(name)
            lo, hi = config.bounds[name]
            v = initial[name]
            fraction = (
                (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
                if name in ("Km_mM", "kcat_per_s", "optical")
                else (v - lo) / (hi - lo)
            )
            fraction = min(max(fraction, 1e-6), 1 - 1e-6)
            self.raw[name] = nn.Parameter(
                torch.tensor(math.log(fraction / (1 - fraction)), dtype=dtype, device=config.device)
            )

    def physical(self):
        values = {}
        for name, (lo, hi) in self.config.bounds.items():
            if name in self.fixed:
                values[name] = torch.tensor(
                    self.fixed[name],
                    dtype=getattr(torch, self.config.dtype),
                    device=self.config.device,
                )
            else:
                frac = torch.sigmoid(self.raw[name])
                values[name] = (
                    torch.exp(math.log(lo) + frac * math.log(hi / lo))
                    if name in ("Km_mM", "kcat_per_s", "optical")
                    else lo + frac * (hi - lo)
                )
        return values

    def signal(self, batch, values=None):
        v = self.physical() if values is None else values
        kinetic_time = torch.clamp(batch["reaction_time"] - v["lag_s"], min=0)
        s = substrate(kinetic_time, batch["s0"], batch["enzyme"], v["Km_mM"], v["kcat_per_s"])
        product = batch["s0"] - s
        alpha = (
            v["optical"] * torch.ones_like(batch["length"])
            if self.alpha_mode
            else epsilon_to_alpha(v["optical"], batch["length"], self.config.reporter_stoichiometry)
        )
        signal = batch["direction"] * alpha * product + v["drift_AU_per_s"] * batch["time"]
        return signal, s, product, alpha

    def evaluate(self, batch, values=None):
        signal, s, product, alpha = self.signal(batch, values)
        mask = batch["mask"]
        baseline = ((batch["observed"] - signal) * mask).sum(1, keepdim=True) / mask.sum(
            1, keepdim=True
        )
        prediction = baseline + signal
        residual = (batch["observed"] - prediction) * mask
        normalized = residual / batch["scale"]
        loss = (normalized.square().sum(1) / mask.sum(1)).mean()
        return loss, prediction, residual, baseline, s, product, alpha


def make_batch(data, indices, direction):
    c = data.config

    def tensor(a):
        return torch.as_tensor(a, dtype=getattr(torch, c.dtype), device=c.device)

    mask = data.used[indices]
    obs = data.signal[indices]
    scales = np.array([max(np.ptp(y[m]), c.scale_floor_AU) for y, m in zip(obs, mask)])
    if c.weighting == "raw_mse":
        scales[:] = 1
    return {
        "time": tensor(data.time[indices]),
        "reaction_time": tensor(data.reaction_time[indices]),
        "observed": tensor(np.where(mask, obs, 0)),
        "mask": tensor(mask),
        "s0": tensor([[data.traces[i]["S0_mM"]] for i in indices]),
        "enzyme": tensor([[data.traces[i]["enzyme_uM"]] for i in indices]),
        "length": tensor(data.lengths[indices, None]),
        "scale": tensor(scales[:, None]),
        "direction": direction,
    }
