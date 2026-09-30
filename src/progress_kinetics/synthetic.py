"""Public fixtures generated independently by SciPy ODE integration."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp


def ode_substrate(time_s, s0_mM, enzyme_uM, km_mM, kcat_per_s):
    """Independent reference: integrate dS/dt with DOP853 from reaction origin."""
    times = np.asarray(time_s, dtype=float)
    if np.any(~np.isfinite(times)) or np.any(times < 0):
        raise ValueError("Times must be finite nonnegative")
    if s0_mM == 0 or enzyme_uM == 0 or times.max(initial=0) == 0:
        return np.full(times.shape, s0_mM)
    solution = solve_ivp(
        lambda t, s: -kcat_per_s * enzyme_uM / 1000 * s / (km_mM + s),
        (0, float(times.max())),
        [s0_mM],
        method="DOP853",
        rtol=2e-11,
        atol=1e-13,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return np.clip(solution.sol(times)[0], 0, s0_mM)


def generate_example(*, seed=7, noise_AU=0.0005, artifacts=True):
    """Two synthetic plates, two enzymes, dilution series, varying E, both signs."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0, 1600, 81)
    substrates = np.tile(np.repeat([0.05, 0.12, 0.3, 0.7, 1.5, 3.0], 2), 2)
    substrates = np.append(substrates, [0, 0.3])
    wells = [f"A{i + 1:02d}" for i in range(len(substrates))]
    shape = (2, len(times), len(wells))
    signal = np.empty(shape)
    saturated = np.zeros(shape, dtype=bool)
    s0 = np.tile(substrates, (2, 1))
    enzymes = rng.uniform(0.25, 0.8, size=s0.shape)
    enzymes[:, -1] = 0
    ids = np.tile(np.array(["SYNTH_A"] * 12 + ["SYNTH_B"] * 12 + ["SYNTH_A"] * 2), (2, 1))
    truth = {
        "SYNTH_A": {
            "Km_mM": 0.3,
            "kcat_per_s": 12,
            "epsilon_per_M_cm": 6000,
            "direction": "decreasing",
        },
        "SYNTH_B": {
            "Km_mM": 0.6,
            "kcat_per_s": 20,
            "epsilon_per_M_cm": 4200,
            "direction": "increasing",
        },
    }
    for p in range(2):
        for w in range(len(wells)):
            info = truth[ids[p, w]]
            s = ode_substrate(times, s0[p, w], enzymes[p, w], info["Km_mM"], info["kcat_per_s"])
            direction = -1 if info["direction"] == "decreasing" else 1
            baseline = 20.0 if direction < 0 else 0.2
            signal[p, :, w] = baseline + direction * info["epsilon_per_M_cm"] / 1000 * (
                s0[p, w] - s
            )
    signal += rng.normal(0, noise_AU, size=shape)
    if artifacts:
        signal[0, 3, 0] = np.nan
        signal[1, :4, 13] = np.nan
        saturated[0, 60, 11] = True
        signal[0, 60, 11] = np.nan
        # Deliberately finite but invalid later readings: the mask must trim them.
        signal[0, 61:, 11] = 100
    ds = xr.Dataset(
        {
            "absorbance_corrected": (("plate_read", "time", "well_384"), signal),
            "saturated": (("plate_read", "time", "well_384"), saturated),
            # Reverse metadata order deliberately exercises named-dimension alignment.
            "cofactor_concentration_mM": (("well_384", "plate_read"), s0.T),
            "expected_conc_um_NORM1": (("plate_read", "well_384"), enzymes),
            "PLASMIDID": (("well_384", "plate_read"), ids.T),
            "mastermix_id": "synthetic_buffer",
            "cofactor_name": "synthetic_substrate",
            "wavelength_nm": 340.0,
            "NaPO3_concentration_mM": 50.0,
            "sampletype": (("plate_read", "well_384"), np.full(s0.shape, "synthetic")),
            "pathlength": (("plate_read", "well_384"), rng.uniform(0.25, 0.5, size=s0.shape)),
        },
        coords={"plate_read": ["plate_1", "plate_2"], "time": times, "well_384": wells},
        attrs={
            "synthetic": "true",
            "reference_path_length_cm": 1.0,
            "truth_json": json.dumps(truth),
            "generator": "independent_scipy_DOP853",
        },
    )
    for name, unit in (
        ("time", "s"),
        ("absorbance_corrected", "AU"),
        ("cofactor_concentration_mM", "mM"),
        ("expected_conc_um_NORM1", "uM"),
        ("pathlength", "cm"),
    ):
        ds[name].attrs["units"] = unit
    return ds
