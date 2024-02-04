from jax.config import config

config.update("jax_enable_x64", True)
import numpy as np
import pytest
from s2ball.transform import laguerre, ball_wavelet
from s2ball.wavelets.helper_functions import *
from s2ball.construct import matrix

L_to_test = [6, 8]
P_to_test = [6, 8]
N_to_test = [1, 2, 3]
lam_to_test = [2.0, 3.0]
tau_to_test = [1.0]
methods_to_test = ["numpy", "jax"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_roundtrip_wavelet(
    flmp_generator, L: int, P: int, N: int, lam: float, tau: float, method: str
):
    Jl = j_max(L, lam)
    Jp = j_max(P, lam)

    flmp = flmp_generator(L, P)
    f = laguerre.inverse(flmp, L, P, tau, method=method)

    matrices = matrix.generate_matrices(
        transform="wavelet", L=L, N=N, P=P, tau=tau, lam_l=lam, lam_p=lam
    )

    w = ball_wavelet.forward(f, L, N, P, matrices, lam, lam, tau, method=method)
    f = ball_wavelet.inverse(w, L, N, P, matrices, lam, lam, tau, method=method)

    w2 = ball_wavelet.forward(f, L, N, P, matrices, lam, lam, tau, method=method)
    f2 = ball_wavelet.inverse(w2, L, N, P, matrices, lam, lam, tau, method=method)

    np.testing.assert_allclose(f, f2, atol=1e-14)
    np.testing.assert_allclose(w[0], w2[0], atol=1e-14)

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            np.testing.assert_allclose(w[1][jp][jl], w2[1][jp][jl], atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_forward_adjoint_wavelet(
    flmp_generator, L: int, P: int, N: int, lam: float, tau: float, method: str
):
    Jl = j_max(L, lam)
    Jp = j_max(P, lam)

    flmp = flmp_generator(L, P)
    f = laguerre.inverse(flmp, L, P, tau, method=method)

    matrices = matrix.generate_matrices(
        transform="wavelet", L=L, N=N, P=P, tau=tau, lam_l=lam, lam_p=lam
    )
    w = ball_wavelet.forward(f, L, N, P, matrices, lam, lam, tau, method=method)
    f2 = ball_wavelet.forward(
        w, L, N, P, matrices, lam, lam, tau, method=method, adjoint=True
    )

    a = np.vdot(f, f2)
    b = np.vdot(w[0], w[0])

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            b += np.vdot(w[1][jp][jl], w[1][jp][jl])

    assert np.abs(a) == pytest.approx(np.abs(b))


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("tau", tau_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_inverse_adjoint_wavelet(
    flmp_generator, L: int, P: int, N: int, lam: float, tau: float, method: str
):
    Jl = j_max(L, lam)
    Jp = j_max(P, lam)

    flmp = flmp_generator(L, P)
    matrices = matrix.generate_matrices(
        transform="wavelet", L=L, N=N, P=P, tau=tau, lam_l=lam, lam_p=lam
    )

    f = laguerre.inverse(flmp, L, P, tau, method=method)
    w = ball_wavelet.forward(f, L, N, P, matrices, lam, lam, tau, method=method)
    f2 = ball_wavelet.inverse(w, L, N, P, matrices, lam, lam, tau, method=method)
    w2 = ball_wavelet.inverse(
        f, L, N, P, matrices, lam, lam, tau, method=method, adjoint=True
    )

    a = np.vdot(f, f2)
    b = np.vdot(w[0], w2[0])

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            b += np.vdot(w[1][jp][jl], w2[1][jp][jl])

    assert np.abs(a) == pytest.approx(np.abs(b))
