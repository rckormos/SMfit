"""Serializable configuration; all concentrations and optical conventions are explicit."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FitConfig:
    plate_dim: str = "plate_read"
    time_dim: str = "time"
    well_dim: str = "well_384"
    signal_var: str = "absorbance_corrected"
    time_var: str = "time"
    substrate_var: str = "cofactor_concentration_mM"
    enzyme_var: str = "expected_conc_um_NORM1"
    identity_var: str = "PLASMIDID"
    group_var: str | None = None
    group_by: tuple[str, ...] | None = None
    merge_plates: bool = False
    single_enzyme: bool = False
    selections: dict[str, list] = field(default_factory=dict)
    sample_role_var: str = "sampletype"
    sample_roles: dict[str, str] = field(default_factory=dict)
    saturation_var: str | None = "saturated"
    saturation_time_var: str | None = "saturation_time_s"
    trim_at_first_missing: bool = False
    units_overrides: dict[str, str] = field(default_factory=dict)
    duplicate_time: str = "reject"
    sort_time: bool = False
    minimum_points: int = 5
    time_window_s: tuple[float | None, float | None] = (None, None)
    signal_direction: str = "auto"
    direction_min_traces: int = 3
    direction_agreement: float = 0.75
    direction_min_change_AU: float = 0.002
    direction_noise_multiplier: float = 3.0
    signal_mode: str = "corrected"
    reference_path_length_cm: float | None = None
    pathlength_var: str = "pathlength"
    fixed_path_length_cm: float | None = None
    effective_alpha_only: bool = False
    reporter_stoichiometry: float = 1.0
    extinction_coefficient_M_cm: float | None = None
    fix_alpha_AU_per_mM: float | None = None
    reaction_age_at_first_read_s: float | dict[str, float] = 0.0
    lag_s: float = 0.0
    fit_lag: bool = False
    fit_drift: bool = False
    enzyme_basis: str = "supplied_concentration"
    starts: dict[str, float] = field(
        default_factory=lambda: {
            "Km_mM": 0.3,
            "kcat_per_s": 10.0,
            "optical": 6000.0,
            "lag_s": 1.0,
            "drift_AU_per_s": 0.0,
        }
    )
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "Km_mM": (1e-5, 100.0),
            "kcat_per_s": (1e-4, 1e5),
            "optical": (1e-3, 1e7),
            "lag_s": (0.0, 600.0),
            "drift_AU_per_s": (-0.1, 0.1),
        }
    )
    fixed_parameters: dict[str, float] = field(default_factory=dict)
    adam_steps: int = 150
    lbfgs_steps: int = 150
    learning_rate: float = 0.05
    tolerance_grad: float = 1e-9
    tolerance_change: float = 1e-12
    multistart: int = 3
    seed: int = 0
    weighting: str = "normalized_mse"
    scale_floor_AU: float = 0.01
    identifiability: str = "sensitivity"
    sensitivity_condition_limit: float = 1e5
    profile_factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    device: str = "cpu"
    dtype: str = "float64"
    plots: bool = True

    def __post_init__(self):
        for key in (
            "minimum_points",
            "multistart",
            "direction_min_traces",
            "adam_steps",
            "lbfgs_steps",
            "seed",
        ):
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        choices = {
            "signal_direction": ("auto", "increasing", "decreasing"),
            "signal_mode": ("corrected", "raw"),
            "duplicate_time": ("reject", "mean"),
            "weighting": ("normalized_mse", "raw_mse"),
            "identifiability": ("sensitivity", "profile"),
            "dtype": ("float64", "float32"),
        }
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key} must be one of {allowed}")
        if self.extinction_coefficient_M_cm is not None and self.fix_alpha_AU_per_mM is not None:
            raise ValueError("Fixed epsilon and fixed alpha are mutually exclusive")
        if self.effective_alpha_only and self.extinction_coefficient_M_cm is not None:
            raise ValueError("Effective-alpha-only mode cannot use fixed epsilon")
        if self.signal_mode == "raw" and self.fix_alpha_AU_per_mM is not None:
            if not self.effective_alpha_only:
                raise ValueError("Fixed alpha in raw mode requires effective_alpha_only=True")
        for key in (
            "reporter_stoichiometry",
            "scale_floor_AU",
            "learning_rate",
            "tolerance_grad",
            "tolerance_change",
            "sensitivity_condition_limit",
        ):
            self._positive(key, getattr(self, key))
        for key in (
            "reference_path_length_cm",
            "fixed_path_length_cm",
            "extinction_coefficient_M_cm",
            "fix_alpha_AU_per_mM",
        ):
            if getattr(self, key) is not None:
                self._positive(key, getattr(self, key))
        if self.minimum_points < 3 or self.multistart < 1 or self.direction_min_traces < 1:
            raise ValueError("Need minimum_points>=3, multistart>=1, direction_min_traces>=1")
        if self.adam_steps < 0 or self.lbfgs_steps < 0:
            raise ValueError("Optimizer step counts must be nonnegative")
        if not 0.5 < self.direction_agreement <= 1:
            raise ValueError("direction_agreement must be >0.5 and <=1")
        for value in (self.lag_s, self.direction_min_change_AU, self.direction_noise_multiplier):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Lag and direction thresholds must be finite nonnegative")
        ages = self.reaction_age_at_first_read_s
        for age in ages.values() if isinstance(ages, dict) else [ages]:
            if not math.isfinite(age) or age < 0:
                raise ValueError("Reaction ages must be finite nonnegative")
        lo, hi = self.time_window_s
        if any(v is not None and not math.isfinite(v) for v in (lo, hi)):
            raise ValueError("Time-window endpoints must be finite")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("Time-window start exceeds end")
        required = {"Km_mM", "kcat_per_s", "optical", "lag_s", "drift_AU_per_s"}
        if set(self.bounds) != required or set(self.starts) != required:
            raise ValueError("Bounds require all five named parameters; unknown start parameter")
        if not set(self.fixed_parameters) <= {"Km_mM", "kcat_per_s"}:
            raise ValueError("fixed_parameters supports Km_mM and kcat_per_s")
        for key, (lower, upper) in self.bounds.items():
            if not math.isfinite(lower + upper) or lower >= upper:
                raise ValueError(f"Invalid bounds for {key}")
            if key in ("Km_mM", "kcat_per_s", "optical") and lower <= 0:
                raise ValueError(f"{key} bounds must be positive")
            if key == "lag_s" and lower < 0:
                raise ValueError("Lag bounds must be nonnegative")
        for key, value in {**self.starts, **self.fixed_parameters}.items():
            lo, hi = self.bounds[key]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{key} value must lie within its bounds")
        for factor in self.profile_factors:
            self._positive("profile factor", factor)
        if self.device == "mps" and self.dtype == "float64":
            raise ValueError("MPS does not support float64; use CPU or explicitly select float32")
        if self.group_by and set(self.group_by) & {
            self.substrate_var,
            self.enzyme_var,
            self.well_dim,
            self.time_dim,
        }:
            raise ValueError("Substrate, enzyme concentration, well and time are not group keys")
        if any(
            role not in ("kinetic", "control", "exclude") for role in self.sample_roles.values()
        ):
            raise ValueError("Sample roles must be kinetic, control, or exclude")

    @staticmethod
    def _positive(name, value):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite positive")

    def to_dict(self):
        return asdict(self)

    def to_json(self, path: str | Path | None = None):
        content = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        if path is not None:
            Path(path).write_text(content + "\n")
        return content

    @classmethod
    def from_json(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text()))
