import numpy as np
import pytest

from baller.transform import laguerre, ball_wavelet, ball_wavelet_adjoint
from baller.wavelets.helper_functions import *

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

    f_wav, f_scal = ball_wavelet.forward(
        f, L, N, P, lam, lam, tau, method=method
    )
    f = ball_wavelet.inverse(
        f_wav, f_scal, L, N, P, lam, lam, tau, method=method
    )

    f_wav_check, f_scal_check = ball_wavelet.forward(
        f, L, N, P, lam, lam, tau, method=method
    )
    f_check = ball_wavelet.inverse(
        f_wav_check, f_scal_check, L, N, P, lam, lam, tau, method=method
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)
    np.testing.assert_allclose(f_scal, f_scal_check, atol=1e-14)

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            np.testing.assert_allclose(
                f_wav[jp][jl], f_wav_check[jp][jl], atol=1e-14
            )


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
    f_wav, f_scal = ball_wavelet.forward(
        f, L, N, P, lam, lam, tau, method=method
    )
    f_adjoint = ball_wavelet_adjoint.forward(
        f_wav, f_scal, L, N, P, lam, lam, tau, method=method
    )

    a = np.vdot(f, f_adjoint)
    b = np.vdot(f_scal, f_scal)

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            b += np.vdot(f_wav[jp][jl], f_wav[jp][jl])

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
    f = laguerre.inverse(flmp, L, P, tau, method=method)
    f_wav, f_scal = ball_wavelet.forward(
        f, L, N, P, lam, lam, tau, method=method
    )

    f_inverse = ball_wavelet.inverse(
        f_wav, f_scal, L, N, P, lam, lam, tau, method=method
    )

    f_wav_adjoint, f_scal_adjoint = ball_wavelet_adjoint.inverse(
        f, L, N, P, lam, lam, tau, method=method
    )

    a = np.vdot(f, f_inverse)
    b = np.vdot(f_scal, f_scal_adjoint)

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            b += np.vdot(f_wav[jp][jl], f_wav_adjoint[jp][jl])

    assert np.abs(a) == pytest.approx(np.abs(b))
