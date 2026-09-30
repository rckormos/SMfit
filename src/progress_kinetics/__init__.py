"""Enzyme-agnostic global progress-curve fitting with labeled xarray data."""

from .api import fit_dataset, validate_dataset
from .config import FitConfig
from .data import SchemaError
from .results import FitResults

__version__ = "0.1.0"
__all__ = ["FitConfig", "FitResults", "SchemaError", "fit_dataset", "validate_dataset"]
