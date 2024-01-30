import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
from s2ball.wavelets import kernels
from s2ball.wavelets.helper_functions import *

from jax.config import config

config.update("jax_enable_x64", True)


def construct_wav_lmp(
    L: int, P: int, lam_l: float, lam_p: float
) -> List[List[np.ndarray]]:
    """Generate multiresolution wavelet filters

    Args:
        L (int): Harmonic band-limit.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.

    Returns:
        List[List[np.ndarray]]: List of wavelet filters for each wavelet scale.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)

    wav_lmp = []
    for jp in range(Jp + 1):
        wav_lmp_entry = []
        for jl in range(Jl + 1):
            L_l, L_p = wavelet_scale_limits(L, P, jl, jp, lam_l, lam_p)
            jl_jp_array = np.zeros((L_p, L_l, 2 * L_l - 1), dtype=np.complex128)
            wav_lmp_entry.append(jl_jp_array)
        wav_lmp.append(wav_lmp_entry)
    return wav_lmp


def construct_f_wav_lmnp(
    L: int, N: int, P: int, lam_l: float, lam_p: float
) -> List[List[np.ndarray]]:
    """Generate multiresolution wavelet Wigner-Laguerre coefficients for Numpy.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.

    Returns:
        List[List[np.ndarray]]: List of wavelet Wigner-Laguerre coefficients.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)

    wav_lmp = []
    for jp in range(Jp + 1):
        wav_lmp_entry = []
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            jl_jp_array = np.zeros(
                (L_p, 2 * Nj - 1, L_l, 2 * L_l - 1), dtype=np.complex128
            )
            wav_lmp_entry.append(jl_jp_array)
        wav_lmp.append(wav_lmp_entry)
    return wav_lmp


def construct_f_wav_lmnp_jax(
    L: int, N: int, P: int, lam_l: float, lam_p: float
) -> List[List[jnp.ndarray]]:
    """Generate multiresolution wavelet Wigner-Laguerre coefficients for JAX.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.

    Returns:
        List[List[jnp.ndarray]]: List of wavelet Wigner-Laguerre coefficients.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)

    wav_lmp = []
    for jp in range(Jp + 1):
        wav_lmp_entry = []
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            jl_jp_array = jnp.zeros(
                (L_p, 2 * Nj - 1, L_l, 2 * L_l - 1), dtype=jnp.complex128
            )
            wav_lmp_entry.append(jl_jp_array)
        wav_lmp.append(wav_lmp_entry)
    return wav_lmp


def construct_f_wav(
    L: int, N: int, P: int, lam_l: float, lam_p: float
) -> List[List[np.ndarray]]:
    """Generate multiresolution wavelet Wigner-Laguerre coefficients for Numpy.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.

    Returns:
        List[List[np.ndarray]]: List of wavelet pixel-space coefficients for each scale.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)

    wav_lmp = []
    for jp in range(Jp + 1):
        wav_lmp_entry = []
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            jl_jp_array = np.zeros(
                (L_p, 2 * Nj - 1, L_l, 2 * L_l - 1), dtype=np.complex128
            )
            wav_lmp_entry.append(jl_jp_array)
        wav_lmp.append(wav_lmp_entry)
    return wav_lmp


def compute_wav_lmp(
    L: int, N: int, P: int, lam_l: float, lam_p: float
) -> Tuple[List[List[np.ndarray]], np.ndarray]:
    """Compute multiresolution wavelet and scaling filters.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.

    Returns:
        Tuple[List[List[np.ndarray]], np.ndarray]: List of wavelet filters for each
            wavelet scale, and an array scaling coefficients.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)

    s_elm = kernels.tiling_direction(L, N)
    kappa_lp, kappa0_lp = tiling_axisym(L, P, lam_l, lam_p)

    wav_lmp = construct_wav_lmp(L, P, lam_l, lam_p)
    factor = np.sqrt((2 * np.arange(L) + 1) / (8.0 * np.pi**2))

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l, L_p = wavelet_scale_limits(L, P, jl, jp, lam_l, lam_p)
            wav_lmp[jp][jl] = np.einsum(
                "pl, lm -> plm",
                kappa_lp[jp, jl, :L_p, :L_l] * factor[:L_l],
                s_elm[:L_l, L - L_l : L - 1 + L_l],
            )

    scal_lmp = kappa0_lp * np.sqrt((2 * np.arange(L) + 1) / (4.0 * np.pi))
    return wav_lmp, scal_lmp.astype(np.complex128)


def tiling_axisym(L: int, P: int, lam_l: float, lam_p: float) -> np.ndarray:
    """Axisymmetric tiling functions"""
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)

    kappa0_lp = np.zeros((P, L), dtype=np.float64)
    kappa_lp = np.zeros((Jp + 1, Jl + 1, P, L), dtype=np.float64)

    phi2_l = kernels.k_lam(L, lam_l)
    phi2_p = kernels.k_lam(P, lam_p)

    temp = np.sqrt(
        phi2_l[0, 0] * phi2_p[1, 0] + (phi2_l[1, 0] - phi2_l[0, 0]) * phi2_p[0, 0]
    )
    kappa0_lp[0, 0] = temp if np.isfinite(temp) and not np.isnan(temp) else 0.0

    for p in range(1, P):
        for l in range(L):
            temp = np.sqrt(phi2_l[0, l])
            kappa0_lp[p, l] = temp if np.isfinite(temp) and not np.isnan(temp) else 0.0

    for p in range(P):
        for l in range(1, L):
            temp = np.sqrt(phi2_p[0, p])
            kappa0_lp[p, l] = temp if np.isfinite(temp) and not np.isnan(temp) else 0.0

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            for p in range(P):
                for l in range(L):
                    temp = np.sqrt(phi2_l[jl + 1, l] - phi2_l[jl, l])
                    temp *= np.sqrt(phi2_p[jp + 1, p] - phi2_p[jp, p])
                    kappa_lp[jp, jl, p, l] = (
                        temp if np.isfinite(temp) and not np.isnan(temp) else 0.0
                    )

    return kappa_lp, kappa0_lp
