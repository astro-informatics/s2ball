import numpy as np
from typing import List
from s2ball.sampling import laguerre_sampling
from s2ball.construct import legendre_constructor, wigner_constructor
from s2ball.wavelets.helper_functions import *


def wavelet_legendre_kernels(L: int, save_dir: str = ".matrices") -> np.ndarray:
    """Constructs an array which holds associated Legendre matrices for the
        forward and inverse spherical harmonic transforms respectively.

    Args:
        L (int): Harmonic band-limit.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".

    Returns:
        np.ndarray: Array of shape [2, L, 2L-1, L] which contains associated Legendre matrices.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    leg_kernel_for = legendre_constructor.load_legendre_matrix(
        L=L, forward=True, spin=0, save_dir=save_dir
    )
    leg_kernel_inv = legendre_constructor.load_legendre_matrix(
        L=L, forward=False, spin=0, save_dir=save_dir
    )
    return np.stack([leg_kernel_for, leg_kernel_inv])


def scaling_laguerre_kernels(P: int, tau: float) -> np.ndarray:
    """Constructs and array which holds the Laguerre polynomials for forward and inverse
        Fourier-Laguerre transforms respectively.

    Args:
        P (int): Radial bandlimit.
        tau (float): Laguerre polynomial scale factor.

    Returns:
        np.ndarray: Array of shape [2, R, P] which contains Laguerre polynomials sampled
            at radial nodes R.
    """
    lag_poly_for = laguerre_sampling.polynomials(P, tau, forward=True)
    lag_poly_inv = laguerre_sampling.polynomials(P, tau, forward=False)
    return np.stack([lag_poly_for, lag_poly_inv])


def wavelet_wigner_kernels(
    L: int,
    N: int,
    lam: float,
    forward: bool = True,
    save_dir: str = ".matrices",
) -> List[np.ndarray]:
    r"""Constructs a collection of Wigner kernels for multiresolution directional wavelet
        transforms.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        lam (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        forward (bool, optional): Whether to load the forward or inverse matrices.
            Defaults to True.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".

    Returns:
        List[np.ndarray]: List of Wigner transform kernels for each angular wavelet scale.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    Jl = j_max(L, lam)
    kernel_list = []

    for jl in range(Jl + 1):
        L_l, L_p, Nj = wavelet_scale_limits_N(L, 1, N, jl, 1, lam, 1.0)
        kernel_list.append(
            wigner_constructor.load_wigner_matrix(
                L=L_l, N=Nj, forward=forward, save_dir=save_dir
            )
        )
    return kernel_list


def wavelet_laguerre_kernels(
    P: int, lam: float, tau: float, forward: bool = True
) -> List[np.ndarray]:
    r"""Constructs a collection of Laguerre polynomial kernel for multiresolution
        directional wavelet transforms.

    Args:
        P (int): Radial band-limit.
        lam (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        tau (float): Laguerre polynomial scale factor.
        forward (bool, optional): Whether to load the forward or inverse matrices.
            Defaults to True.

    Returns:
        List[np.ndarray]: List of Laguerre polynomial kernels for each radial wavelet scale.
    """
    Jp = j_max(P, lam)
    kernel_list = []

    for jp in range(Jp + 1):
        L_l, L_p, Nj = wavelet_scale_limits_N(1, P, 1, 1, jp, 1.0, lam)
        kernel_list.append(laguerre_sampling.polynomials(L_p, tau, forward=forward))
    return kernel_list
