from jax.config import config

config.update("jax_enable_x64", True)
import numpy as np
import pytest
from s2ball.transform import wigner_laguerre as laguerre
from s2ball.construct import matrix

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
    matrices = matrix.generate_matrices(
        transform="wigner_laguerre", L=L, P=P, N=N, tau=tau
    )
    flmnp = flmnp_generator(L, N, P)

    f = laguerre.inverse(flmnp, L, N, P, tau, matrices, method)
    flmnp = laguerre.forward(f, L, N, P, tau, matrices, method)

    f_check = laguerre.inverse(flmnp, L, N, P, tau, matrices, method)
    flmnp_check = laguerre.forward(f_check, L, N, P, tau, matrices, method)

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
