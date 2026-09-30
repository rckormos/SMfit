"""Labeled results and reproducible, safe CSV/JSON/NetCDF/Zarr exports."""

from __future__ import annotations

import csv
import datetime
import importlib.metadata
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from .data import stable


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return stable(value)


def safe_dataset(dataset):
    """Return a copy with portable object variables and NetCDF-safe attributes."""
    ds = dataset.copy(deep=True)
    for name in list(ds.variables):
        da = ds[name]
        attrs = dict(da.attrs)
        if da.dtype.kind == "O":
            values = np.array([stable(v) for v in da.values.ravel()], dtype=str).reshape(da.shape)
            if name in ds.coords:
                ds = ds.assign_coords({name: (da.dims, values)})
            else:
                ds[name] = (da.dims, values)
        ds[name].attrs = _safe_attrs(attrs)
        ds[name].encoding = {}
    ds.attrs = _safe_attrs(ds.attrs)
    return ds


def _safe_attrs(attrs):
    return {
        str(k): (
            v
            if isinstance(v, (str, int, float)) and not isinstance(v, bool)
            else json.dumps(clean_json(v), sort_keys=True)
        )
        for k, v in attrs.items()
    }


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(clean_json(value), sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def write_zarr(dataset, path):
    """Write portable format-2 stores across the supported xarray versions."""
    keyword = (
        "zarr_format"
        if "zarr_format" in inspect.signature(dataset.to_zarr).parameters
        else "zarr_version"
    )
    return dataset.to_zarr(path, mode="w", **{keyword: 2})


@dataclass
class FitResults:
    parameters: xr.Dataset
    predictions: xr.Dataset
    diagnostics: xr.Dataset
    parameter_rows: list[dict]
    trace_rows: list[dict]
    run: dict

    @property
    def exit_code(self):
        good = sum(row["status"] in ("success", "partial_success") for row in self.parameter_rows)
        if not good:
            return 4
        return 0 if all(row["status"] == "success" for row in self.parameter_rows) else 3

    def write(self, outdir, *, format="netcdf", plots=None):
        if format not in ("netcdf", "zarr"):
            raise ValueError("Output format must be netcdf or zarr")
        root = Path(outdir)
        root.mkdir(parents=True, exist_ok=True)
        write_csv(root / "fit_parameters.csv", self.parameter_rows)
        write_csv(root / "trace_diagnostics.csv", self.trace_rows)
        run = {**self.run, "exit_code": self.exit_code, "output_format": format}
        (root / "run.json").write_text(
            json.dumps(clean_json(run), indent=2, allow_nan=False) + "\n"
        )
        (root / "config.json").write_text(json.dumps(clean_json(run["config"]), indent=2) + "\n")
        combined = xr.merge(
            [self.predictions, self.parameters, self.diagnostics], compat="no_conflicts"
        )
        combined.attrs.update(
            {
                "software": "progress-kinetics",
                "schema_version": "1",
                "conventions_json": json.dumps(clean_json(run["conventions"])),
            }
        )
        combined = safe_dataset(combined)
        if format == "netcdf":
            combined.to_netcdf(root / "fit_results.nc", engine="h5netcdf")
        else:
            write_zarr(combined, root / "fit_results.zarr")
        if plots is None:
            plots = run["config"]["plots"]
        if plots:
            from .plotting import plot_results

            plot_results(self, root)
        return root


def versions():
    names = (
        "progress-kinetics",
        "numpy",
        "torch",
        "xarray",
        "scipy",
        "matplotlib",
        "h5netcdf",
        "zarr",
    )
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "source_tree"
    return result
