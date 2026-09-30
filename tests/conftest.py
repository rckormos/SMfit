import pytest

from progress_kinetics import FitConfig, fit_dataset
from progress_kinetics.synthetic import generate_example


@pytest.fixture(scope="session")
def example():
    return generate_example(noise_AU=0, artifacts=False)


@pytest.fixture
def single(example):
    return example.isel(plate_read=[0], well_384=slice(0, 12)).copy(deep=True)


@pytest.fixture(scope="session")
def favorable_fit(example):
    ds = example.isel(plate_read=[0], well_384=slice(0, 12))
    return fit_dataset(ds, FitConfig(adam_steps=30, lbfgs_steps=100, multistart=2, plots=False))
