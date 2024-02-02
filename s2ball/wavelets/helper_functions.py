import numpy as np
import math
from typing import Tuple


def j_max(L: int, lam: float = 2) -> int:
    """Computes the highest wavelet scale :math:`J_{\text{max}}`.
    Args:
        L (int): Harmonic band-limit.
        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
    Returns:
        int: Maximum wavelet scale :math:`J_{\text{max}}`.
    """
    return math.ceil(np.log(L) / np.log(lam))


def radial_bandlimit(jp: int, lam_p: float = 2) -> int:
    """Computes the radial band-limit for scale :math:`j_p`.
    Args:
        jp (int): Wavelet radial scale.
        lam_p (float, optional): Radial wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
    Returns:
        int: Radial band-limit for scale :math:`j_p`.
    """
    return math.ceil(lam_p ** (jp + 1))


def angular_bandlimit(jl: int, lam_l: float) -> int:
    """Computes the angular band-limit for scale :math:`j_{\ell}`.
    Args:
        jl (int): Wavelet angular scale.
        lam_l (float, optional): Angular wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
    Returns:
        int: Angular band-limit for scale :math:`j_l`.
    """
    return math.ceil(lam_l ** (jl + 1))


def wavelet_scale_limits(
    L: int, P: int, jl: int, jp: int, lam_l: float = 2, lam_p: float = 2
) -> Tuple[int, int]:
    """Computes the angular and radial band-limits for scale :math:`j_{\ell}/j_p`.
    Args:
        L (int): Harmonic band-limit.
        P (int): Radial band-limit.
        jl (int): Wavelet angular scale.
        jp (int): Wavelet radial scale.
        lam_l (float, optional): Angular wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
        lam_p (float, optional): Radial wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
    Returns:
        Tuple[int,int]: Angular and Radial band-limit for scale :math:`j_{\ell}/j_p`.
    """
    return min(angular_bandlimit(jl, lam_l), L), min(radial_bandlimit(jp, lam_p), P)


def wavelet_scale_limits_N(
    L: int, P: int, N: int, jl: int, jp: int, lam_l: float = 2, lam_p: float = 2
) -> Tuple[int, int, int]:
    """Computes the angular and radial band-limits and multiresolution directionality :math:`N_j` for scale :math:`j_{\ell}/j_p`.
    Args:
        L (int): Harmonic band-limit.
        P (int): Radial band-limit.
        N (int): Azimuthal (directional) band-limit.
        jl (int): Wavelet angular scale.
        jp (int): Wavelet radial scale.
        lam_l (float, optional): Angular wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
        lam_p (float, optional): Radial wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
    Returns:
        Tuple[int,int]: Angular and Radial band-limit for scale :math:`j_{\ell}/j_p`.
    """
    L_l = min(angular_bandlimit(jl, lam_l), L)
    L_p = min(radial_bandlimit(jp, lam_p), P)
    Nj = min(N, L_l)
    Nj += (Nj + N) % 2
    return L_l, L_p, Nj
