import numpy as np
import pytest

from s2ball.transform import wigner_laguerre as laguerre
from s2ball.construct.wigner_constructor import *
from s2ball.sampling import laguerre_sampling

L_to_test = [8, 12, 16]
N_to_test = [2, 4, 6]
P_to_test = [8, 12, 16]
tau_to_test = [1.0, 2.0]
methods_to_test = ["numpy", "jax"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_roundtrip_wigner_laguerre(
    flmnp_generator, L: int, N: int, P: int, tau: float, method: str
):
    wigner_forward = load_wigner_matrix(L, N, forward=True)
    wigner_inverse = load_wigner_matrix(L, N, forward=False)

    lag_poly_f = laguerre_sampling.polynomials(P, tau, forward=True)
    lag_poly_i = laguerre_sampling.polynomials(P, tau, forward=False)

    flmnp = flmnp_generator(L, N, P)

    f = laguerre.inverse(flmnp, L, N, P, tau, wigner_inverse, lag_poly_i, method)
    flmnp = laguerre.forward(f, L, N, P, tau, wigner_forward, lag_poly_f, method)

    f_check = laguerre.inverse(flmnp, L, N, P, tau, wigner_inverse, lag_poly_i, method)
    flmnp_check = laguerre.forward(
        f_check, L, N, P, tau, wigner_forward, lag_poly_f, method
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)
    np.testing.assert_allclose(flmnp, flmnp_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_roundtrip_wigner_laguerre_no_kernels(
    flmnp_generator, L: int, N: int, P: int, tau: float, method: str
):
    flmnp = flmnp_generator(L, N, P)

    f = laguerre.inverse(flmnp, L, N, P, tau, method=method)
    flmnp = laguerre.forward(f, L, N, P, tau, method=method)

    f_check = laguerre.inverse(flmnp, L, N, P, tau, method=method)
    flmnp_check = laguerre.forward(f_check, L, N, P, tau, method=method)

    np.testing.assert_allclose(f, f_check, atol=1e-14)
    np.testing.assert_allclose(flmnp, flmnp_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_forward_wigner_laguerre_adjoint(
    flmnp_generator, L: int, N: int, P: int, tau: float, method: str
):
    flmnp = flmnp_generator(L, N, P)
    f = laguerre.inverse(flmnp, L, N, P, tau, method=method)

    flmnp_forward = laguerre.forward(f, L, N, P, tau, method=method)
    f_adjoint = laguerre.forward(flmnp, L, N, P, tau, method=method, adjoint=True)

    a = np.abs(np.vdot(f, f_adjoint))
    b = np.abs(np.vdot(flmnp, flmnp_forward))
    assert a == pytest.approx(b)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_inverse_wigner_laguerre_adjoint(
    flmnp_generator, L: int, N: int, P: int, tau: float, method: str
):
    flmnp = flmnp_generator(L, N, P)
    f = laguerre.inverse(flmnp, L, N, P, tau, method=method)

    flmnp_inverse_adjoint = laguerre.inverse(
        f, L, N, P, tau, method=method, adjoint=True
    )
    f_inverse = laguerre.inverse(flmnp, L, N, P, tau, method=method)

    a = np.abs(np.vdot(f, f_inverse))
    b = np.abs(np.vdot(flmnp, flmnp_inverse_adjoint))
    assert a == pytest.approx(b)
