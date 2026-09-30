"""CLI for serialized labeled data. No pickle or proprietary importer."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import xarray as xr

from .api import fit_dataset, validate_dataset
from .config import FitConfig
from .results import clean_json, safe_dataset, write_zarr


def parser():
    p = argparse.ArgumentParser(
        prog="progress-fit",
        description="Fit shared Km (mM), kcat (s^-1) and optical scale per enzyme/condition group. "
        "Input is an xarray NetCDF or Zarr dataset; corrected absorbance remains AU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="NetCDF file or Zarr store")
    p.add_argument("--outdir", default="results", help="Output directory")
    p.add_argument("--config", help="FitConfig JSON; explicit CLI arguments override its fields")
    p.add_argument("--input-format", choices=("auto", "netcdf", "zarr"), default="auto")
    p.add_argument("--output-format", choices=("netcdf", "zarr"), default="netcdf")
    p.add_argument(
        "--validate-only", action="store_true", help="Inspect schema/QC without optimization"
    )
    p.add_argument(
        "--report-json", help="Write validation summary or invalid-input report to this file"
    )
    defaults = FitConfig()

    def opt(name, *, type=None, choices=None, help=""):
        p.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            type=type,
            choices=choices,
            default=argparse.SUPPRESS,
            help=help + f" (default: {getattr(defaults, name)})",
        )

    for name in (
        "plate_dim",
        "time_dim",
        "well_dim",
        "signal_var",
        "time_var",
        "substrate_var",
        "enzyme_var",
        "identity_var",
        "group_var",
        "pathlength_var",
        "saturation_var",
        "saturation_time_var",
        "sample_role_var",
        "enzyme_basis",
        "device",
    ):
        opt(name, help="Mapped dimension/variable or convention; use config for complex mappings")
    p.add_argument(
        "--absorbance-var",
        dest="signal_var",
        default=argparse.SUPPRESS,
        help="Alias for --signal-var",
    )
    for name in (
        "merge_plates",
        "single_enzyme",
        "trim_at_first_missing",
        "sort_time",
        "effective_alpha_only",
        "fit_lag",
        "fit_drift",
    ):
        p.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            action="store_true",
            default=argparse.SUPPRESS,
            help=f"Explicitly enable {name}; off by default",
        )
    p.add_argument("--no-plots", dest="plots", action="store_false", default=argparse.SUPPRESS)
    for name, choices in (
        ("signal_direction", ("auto", "increasing", "decreasing")),
        ("signal_mode", ("corrected", "raw")),
        ("duplicate_time", ("reject", "mean")),
        ("weighting", ("normalized_mse", "raw_mse")),
        ("identifiability", ("sensitivity", "profile")),
        ("dtype", ("float64", "float32")),
    ):
        opt(name, choices=choices, help="Explicit fit/preprocessing policy")
    for name in (
        "minimum_points",
        "direction_min_traces",
        "adam_steps",
        "lbfgs_steps",
        "multistart",
        "seed",
    ):
        opt(name, type=int, help="Count/seed for validation or optimization")
    units = {
        "reference_path_length_cm": "Normalized reader reference length in cm; measured length is not reapplied",
        "fixed_path_length_cm": "Fixed raw AU optical length in cm",
        "reporter_stoichiometry": "Reporter equivalents per substrate; fixed positive value",
        "extinction_coefficient_M_cm": "Fix positive effective extinction change in M^-1 cm^-1",
        "fix_alpha_AU_per_mM": "Fix effective signal scale in AU/mM; exclusive with fixed epsilon",
        "reaction_age_at_first_read_s": "Known pre-acquisition reaction age in seconds; per-plate mapping in config",
        "lag_s": "Fixed lag in seconds; default zero; --fit-lag explicitly makes it free",
        "scale_floor_AU": "Normalization floor in AU",
        "direction_min_change_AU": "Minimum robust early/late AU change for a sign vote",
    }
    for name in (
        *units,
        "learning_rate",
        "tolerance_grad",
        "tolerance_change",
        "direction_agreement",
        "direction_noise_multiplier",
        "sensitivity_condition_limit",
    ):
        opt(name, type=float, help=units.get(name, "Optimizer/diagnostic threshold"))
    p.add_argument(
        "--group-by",
        default=argparse.SUPPRESS,
        help="Comma-separated metadata; enzyme identity is always checked; no S0/E keys",
    )
    p.add_argument(
        "--time-window",
        nargs=2,
        type=float,
        dest="time_window_s",
        default=argparse.SUPPRESS,
        metavar=("START_S", "END_S"),
    )
    p.add_argument(
        "--profile-factors",
        nargs="+",
        type=float,
        default=argparse.SUPPRESS,
        help="Fixed optical factors, each with reoptimized kinetics; requires --identifiability profile",
    )
    for flag, help_text in (
        ("select", "VAR=value1,value2; repeat for explicit metadata filters"),
        ("sample-role", "LABEL=kinetic|control|exclude; repeat"),
        ("units", "VAR=unit; repeat to override unknown/missing units"),
        ("start", "PARAM=value; repeat; PARAM is Km_mM,kcat_per_s,optical,lag_s,drift_AU_per_s"),
        ("fix-parameter", "Km_mM=value or kcat_per_s=value; repeat"),
    ):
        p.add_argument("--" + flag, action="append", default=[], help=help_text)
    p.add_argument(
        "--bound",
        action="append",
        nargs=3,
        default=[],
        metavar=("PARAM", "LOW", "HIGH"),
        help="Override a named parameter's finite bounds; repeat",
    )
    return p


def load_dataset(path, format="auto"):
    path = Path(path)
    if format == "auto":
        format = "zarr" if path.suffix == ".zarr" or path.is_dir() else "netcdf"
    if format == "zarr":
        with xr.open_zarr(path) as ds:
            return ds.load()
    with xr.open_dataset(path) as ds:
        return ds.load()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "example":
        p = argparse.ArgumentParser(
            prog="progress-fit example", description="Generate only synthetic data"
        )
        p.add_argument("output", help=".nc file or .zarr directory")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--noise-AU", type=float, default=0.0005)
        args = p.parse_args(argv[1:])
        from .synthetic import generate_example

        ds = safe_dataset(generate_example(seed=args.seed, noise_AU=args.noise_AU))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".zarr":
            write_zarr(ds, output)
        else:
            ds.to_netcdf(output, engine="h5netcdf")
        print(f"Wrote synthetic example to {output}")
        return 0
    if argv and argv[0] == "validate":
        argv = argv[1:] + ["--validate-only"]
    p = parser()
    args = vars(p.parse_args(argv))
    try:
        config_path = args.get("config")
        settings = (
            FitConfig.from_json(config_path).to_dict() if config_path else FitConfig().to_dict()
        )
        valid_fields = {f.name for f in fields(FitConfig)}
        for name, value in args.items():
            if name in valid_fields:
                settings[name] = value
        if isinstance(settings["group_by"], str):
            settings["group_by"] = tuple(
                x.strip() for x in settings["group_by"].split(",") if x.strip()
            )
        for source, target, cast in (
            ("select", "selections", lambda v: v.split(",")),
            ("sample_role", "sample_roles", str),
            ("units", "units_overrides", str),
            ("start", "starts", float),
            ("fix_parameter", "fixed_parameters", float),
        ):
            for item in args[source]:
                key, value = item.split("=", 1)
                settings[target][key] = cast(value)
        for key, lo, hi in args["bound"]:
            settings["bounds"][key] = (float(lo), float(hi))
        c = FitConfig(**settings)
        ds = load_dataset(args["input"], args["input_format"])
        if args["validate_only"]:
            report = validate_dataset(ds, c)
            code = 0 if any(row["kinetic_trace_count"] for row in report["groups"]) else 4
            report["exit_code"] = code
        else:
            result = fit_dataset(ds, c)
            result.run["input_path"] = str(Path(args["input"]).resolve())
            result.write(args["outdir"], format=args["output_format"])
            code = result.exit_code
            report = {
                "status": result.run["status"],
                "exit_code": code,
                "outdir": args["outdir"],
                "groups": result.parameter_rows,
                "warnings": result.run["warnings"],
            }
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        report, code = {"status": "invalid_input", "error": str(exc), "exit_code": 2}, 2
    content = json.dumps(clean_json(report), indent=2, allow_nan=False)
    print(content, file=sys.stderr if code == 2 else sys.stdout)
    if args.get("report_json"):
        report_path = Path(args["report_json"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(content + "\n")
    return code
