import numpy as np
from typing import List
from s2ball.sampling import laguerre_sampling
from s2ball.construct import wigner_constructor
from s2ball.wavelets.helper_functions import *


def wavelet_wigner_kernels(
    L: int,
    N: int,
    lam: float,
    save_dir: str = ".matrices",
) -> List[List[np.ndarray]]:
    r"""Constructs a collection of Wigner kernels for multiresolution directional wavelet
        transforms.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        lam (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".

    Returns:
        List[List[np.ndarray]]: List of Wigner transform kernels for each angular wavelet scale.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    Jl = j_max(L, lam)
    kernel_list = []

    for jl in range(Jl + 1):
        L_l, L_p, Nj = wavelet_scale_limits_N(L, 1, N, jl, 1, lam, 1.0)
        scale_entry = []
        scale_entry.append(
            wigner_constructor.load_wigner_matrix(
                L=L_l, N=Nj, forward=True, save_dir=save_dir
            )
        )
        scale_entry.append(
            wigner_constructor.load_wigner_matrix(
                L=L_l, N=Nj, forward=False, save_dir=save_dir
            )
        )
        kernel_list.append(scale_entry)
    return kernel_list


def wavelet_laguerre_kernels(P: int, lam: float, tau: float) -> List[List[np.ndarray]]:
    r"""Constructs a collection of Laguerre polynomial kernel for multiresolution
        directional wavelet transforms.

    Args:
        P (int): Radial band-limit.
        lam (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        tau (float): Laguerre polynomial scale factor.

    Returns:
        List[List[np.ndarray]]: List of Laguerre polynomial kernels for each radial wavelet scale.
    """
    Jp = j_max(P, lam)
    kernel_list = []

    for jp in range(Jp + 1):
        L_l, L_p, Nj = wavelet_scale_limits_N(1, P, 1, 1, jp, 1.0, lam)
        scale_entry = []
        scale_entry.append(laguerre_sampling.polynomials(L_p, tau, forward=True))
        scale_entry.append(laguerre_sampling.polynomials(L_p, tau, forward=False))
        kernel_list.append(scale_entry)
    return kernel_list
