import numpy as np
import pytest

from s2ball.transform import laguerre
from s2ball.construct.legendre_constructor import *
from s2ball.sampling import laguerre_sampling

L_to_test = [8, 12, 16]
P_to_test = [8, 12, 16]
tau_to_test = [1.0, 2.0]
methods_to_test = ["numpy", "jax"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_roundtrip_laguerre(flmp_generator, L: int, P: int, tau: float, method: str):
    legendre_forward = load_legendre_matrix(L, forward=True)
    legendre_inverse = load_legendre_matrix(L, forward=False)

    lag_poly_f = laguerre_sampling.polynomials(P, tau, forward=True)
    lag_poly_i = laguerre_sampling.polynomials(P, tau, forward=False)

    flmp = flmp_generator(L, P)

    f = laguerre.inverse(flmp, L, P, tau, legendre_inverse, lag_poly_i, method)
    flmp = laguerre.forward(f, L, P, tau, legendre_forward, lag_poly_f, method)

    f_check = laguerre.inverse(flmp, L, P, tau, legendre_inverse, lag_poly_i, method)
    flmp_check = laguerre.forward(
        f_check, L, P, tau, legendre_forward, lag_poly_f, method
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)
    np.testing.assert_allclose(flmp, flmp_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_roundtrip_laguerre_no_kernels(
    flmp_generator, L: int, P: int, tau: float, method: str
):
    flmp = flmp_generator(L, P)

    f = laguerre.inverse(flmp, L, P, tau, method=method)
    flmp = laguerre.forward(f, L, P, tau, method=method)

    f_check = laguerre.inverse(flmp, L, P, tau, method=method)
    flmp_check = laguerre.forward(f_check, L, P, tau, method=method)

    np.testing.assert_allclose(f, f_check, atol=1e-14)
    np.testing.assert_allclose(flmp, flmp_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_forward_laguerre_adjoint(
    flmp_generator, L: int, P: int, tau: float, method: str
):
    flmp = flmp_generator(L, P)
    f = laguerre.inverse(flmp, L, P, tau, method=method)

    flmp_forward = laguerre.forward(f, L, P, tau, method=method)
    f_adjoint = laguerre.forward(flmp, L, P, tau, method=method, adjoint=True)

    a = np.abs(np.vdot(f, f_adjoint))
    b = np.abs(np.vdot(flmp, flmp_forward))
    assert a == pytest.approx(b)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_inverse_laguerre_adjoint(
    flmp_generator, L: int, P: int, tau: float, method: str
):
    flmp = flmp_generator(L, P)
    f = laguerre.inverse(flmp, L, P, tau, method=method)

    flmp_inverse_adjoint = laguerre.inverse(f, L, P, tau, method=method, adjoint=True)
    f_inverse = laguerre.inverse(flmp, L, P, tau, method=method)

    a = np.abs(np.vdot(f, f_inverse))
    b = np.abs(np.vdot(flmp, flmp_inverse_adjoint))
    assert a == pytest.approx(b)
