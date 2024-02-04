from jax.config import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
from s2ball.transform import laguerre
from s2ball.construct import matrix

L_to_test = [8, 12, 16]
P_to_test = [8, 12, 16]
tau_to_test = [1.0, 2.0]
methods_to_test = ["numpy", "jax"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_roundtrip_laguerre(flmp_generator, L: int, P: int, tau: float, method: str):

    matrices = matrix.generate_matrices(
        transform="spherical_laguerre", L=L, P=P, tau=tau
    )
    flmp = flmp_generator(L, P)

    f = laguerre.inverse(flmp, L, P, tau, matrices, method)
    flmp = laguerre.forward(f, L, P, tau, matrices, method)

    f_check = laguerre.inverse(flmp, L, P, tau, matrices, method)
    flmp_check = laguerre.forward(f_check, L, P, tau, matrices, method)

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
