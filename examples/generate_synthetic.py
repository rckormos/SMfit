"""Run from the repository after installing: python examples/generate_synthetic.py."""

from pathlib import Path

from progress_kinetics.results import safe_dataset
from progress_kinetics.synthetic import generate_example

output = Path("examples/synthetic.nc")
safe_dataset(generate_example()).to_netcdf(output, engine="h5netcdf")
print(f"Created {output}; all identifiers and measurements are synthetic.")
