import numpy as np
import pytest
import torch

from progress_kinetics.model import epsilon_to_alpha, substrate
from progress_kinetics.schnell_mendoza import product_concentration, substrate_concentration
from progress_kinetics.synthetic import ode_substrate


def test_independent_ode_and_integrated_equation():
    t = torch.linspace(0, 1000, 71, dtype=torch.float64)
    s = substrate(t, 1.5, 0.4, 0.3, 12)
    reference = ode_substrate(t.numpy(), 1.5, 0.4, 0.3, 12)
    np.testing.assert_allclose(s.numpy(), reference, rtol=1e-7, atol=1e-10)
    lhs = s - 1.5 + 0.3 * torch.log(s / 1.5) + 12 * 0.4 / 1000 * t
    assert lhs.abs().max() < 1e-9


def test_per_well_enzyme_and_exact_conversion():
    t = torch.tensor([[0.0, 1.0, 100.0], [0.0, 1.0, 100.0]], dtype=torch.float64)
    enzymes = torch.tensor([[0.1], [0.8]], dtype=torch.float64)
    s = substrate(t, 1.0, enzymes, 0.2, 15.0)
    expected = substrate_concentration(t, 1.0, 15 * enzymes / 1000, 0.2, iterations=24)
    torch.testing.assert_close(s, expected, rtol=0, atol=0)
    assert not torch.allclose(s[0], s[1])
    wrong = substrate(t, 1.0, enzymes.mean(), 0.2, 15.0)
    assert not torch.allclose(s, wrong)


def test_optical_conversion_independent_units():
    # 6000 M^-1 cm^-1 * 0.4 cm * 2 equivalents * 0.001 M per mM
    assert epsilon_to_alpha(6000.0, 0.4, 2.0) == 4.8


def test_exact_zero_controls_and_gradients():
    t = torch.linspace(0, 20000, 100, dtype=torch.float64).reshape(1, -1)
    s0 = torch.tensor([[0.0], [0.5], [0.05]], dtype=torch.float64)
    e = torch.tensor([[0.3], [0.0], [10.0]], dtype=torch.float64)
    km = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    s = substrate(t, s0, e, km, 100.0)
    assert torch.all(s[0] == 0) and torch.all(s[1] == 0.5)
    s.sum().backward()
    assert torch.isfinite(km.grad)
    assert torch.all(s >= 0) and torch.all(s <= s0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_inputs_rejected(bad):
    with pytest.raises(ValueError):
        substrate(torch.tensor([0.0, 1.0], dtype=torch.float64), bad, 1.0, 0.3, 12.0)


def test_donor_port_gradients_and_broadcasting():
    time = torch.linspace(0, 20000, 200, dtype=torch.float64)
    vmax = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    km = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    product = product_concentration(time, 0.05, vmax, km)
    product.mean().backward()
    assert torch.isfinite(vmax.grad) and torch.isfinite(km.grad)
    assert (product >= 0).all() and (product <= 0.05).all()
    t = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    s0 = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    s = substrate_concentration(t, s0, 0.1, 0.3)
    assert s.shape == (2, 2)
    torch.testing.assert_close(s[:, 0], s0[:, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware not available")
def test_cuda_matches_float64_cpu():
    t = torch.linspace(0, 1000, 71, dtype=torch.float64)
    reference = substrate(t, 1.5, 0.4, 0.3, 12)
    actual = substrate(t.cuda(), 1.5, 0.4, 0.3, 12).cpu()
    torch.testing.assert_close(reference, actual, rtol=1e-8, atol=1e-11)
