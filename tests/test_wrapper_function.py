from jax.config import config

config.update("jax_enable_x64", True)
import numpy as np
import pytest
from s2ball.sampling import laguerre_sampling
from s2ball.construct import *

L_to_test = [8, 12, 16]
P_to_test = [8, 12, 16]
N_to_test = [2, 4, 6]
spin_to_test = [-2, 0, 2]
tau_to_test = [1.0, 2.0]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
def test_harmonic_wrapper(L: int, spin: int):
    """Test wrapper for generating Legendre matrices"""

    automatic = matrix.generate_matrices("spherical_harmonic", L=L, spin=spin)
    manual_forward = legendre_constructor.load_legendre_matrix(
        L=L, spin=spin, forward=True
    )
    manual_inverse = legendre_constructor.load_legendre_matrix(
        L=L, spin=spin, forward=False
    )
    assert np.allclose(automatic[0], manual_forward)
    assert np.allclose(automatic[1], manual_inverse)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
def test_wigner_wrapper(L: int, N: int):
    """Test wrapper for generating Wigner matrices"""

    automatic = matrix.generate_matrices("wigner", L=L, N=N)
    manual_forward = wigner_constructor.load_wigner_matrix(L=L, N=N, forward=True)
    manual_inverse = wigner_constructor.load_wigner_matrix(L=L, N=N, forward=False)
    assert np.allclose(automatic[0], manual_forward)
    assert np.allclose(automatic[1], manual_inverse)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
def test_spherical_laguerre_wrapper(L: int, spin: int, P: int, tau: float):
    """Test wrapper for generating Spherical-Laguerre matrices"""

    automatic = matrix.generate_matrices(
        "spherical_laguerre", L=L, P=P, spin=spin, tau=tau
    )
    # Check Legendre matrices
    manual_legendre_forward = legendre_constructor.load_legendre_matrix(
        L=L, spin=spin, forward=True
    )
    manual_legendre_inverse = legendre_constructor.load_legendre_matrix(
        L=L, spin=spin, forward=False
    )
    assert np.allclose(automatic[0][0], manual_legendre_forward)
    assert np.allclose(automatic[0][1], manual_legendre_inverse)

    # Check Laguerre matrices
    manual_laguerre_forward = laguerre_sampling.polynomials(P, tau, forward=True)
    manual_laguerre_inverse = laguerre_sampling.polynomials(P, tau, forward=False)
    assert np.allclose(automatic[1][0], manual_laguerre_forward)
    assert np.allclose(automatic[1][1], manual_laguerre_inverse)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
def test_wigner_laguerre_wrapper(L: int, N: int, P: int, tau: float):
    """Test wrapper for generating Wigner-Laguerre matrices"""

    automatic = matrix.generate_matrices("wigner_laguerre", L=L, P=P, N=N, tau=tau)
    # Check Legendre matrices
    manual_wigner_forward = wigner_constructor.load_wigner_matrix(
        L=L, N=N, forward=True
    )
    manual_wigner_inverse = wigner_constructor.load_wigner_matrix(
        L=L, N=N, forward=False
    )
    assert np.allclose(automatic[0][0], manual_wigner_forward)
    assert np.allclose(automatic[0][1], manual_wigner_inverse)

    # Check Laguerre matrices
    manual_laguerre_forward = laguerre_sampling.polynomials(P, tau, forward=True)
    manual_laguerre_inverse = laguerre_sampling.polynomials(P, tau, forward=False)
    assert np.allclose(automatic[1][0], manual_laguerre_forward)
    assert np.allclose(automatic[1][1], manual_laguerre_inverse)
