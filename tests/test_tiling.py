import numpy as np
import pytest
from baller.wavelets import tiling
from baller.wavelets.helper_functions import *

L_to_test = [8, 12, 16]
P_to_test = [8, 12, 16]
N_to_test = [2, 4]
lam_to_test = [2.0, 3.0]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_laguerre_wavelet_tiling(L: int, P: int, lam: float):
    kappa_lp, kappa0_lp = tiling.tiling_axisym(L, P, lam, lam)
    identity = np.sum(kappa_lp**2) + np.sum(kappa0_lp**2)
    np.isclose(identity, 1)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("P", P_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_laguerre_wavelet_admissibility(L: int, P: int, N: int, lam: float):
    Jl = j_max(L, lam)
    Jp = j_max(P, lam)

    wav_lmp, scal_lmp = tiling.compute_wav_lmp(L, N, P, lam, lam)
    ident = (scal_lmp * np.conj(scal_lmp)) * (
        4.0 * np.pi / (2 * np.arange(L) + 1)
    )

    factor = 8.0 * np.pi**2 / (2 * np.arange(L) + 1)
    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l, L_p = wavelet_scale_limits(L, P, jl, jp, lam, lam)
            jp_jl = wav_lmp[jp][jl] * np.conj(wav_lmp[jp][jl])
            jp_jl = np.einsum("plm,l->plm", jp_jl, factor[:L_l])
            ident[:L_p, :L_l] += np.sum(jp_jl[:L_p, :L_l], axis=-1)

    np.testing.assert_allclose(ident, 1)
