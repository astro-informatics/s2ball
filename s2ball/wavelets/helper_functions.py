import numpy as np
import math
from typing import Tuple


def j_max(L: int, lam: float) -> int:
    """Computes the highest wavelet scale Jmax"""
    return math.ceil(np.log(L) / np.log(lam))


def radial_bandlimit(jp: int, lam_p: float) -> int:
    """computes the radial band-limit for scale jp"""
    return math.ceil(lam_p ** (jp + 1))


def angular_bandlimit(jl: int, lam_l: float) -> int:
    """computes the angular band-limit for scale jl"""
    return math.ceil(lam_l ** (jl + 1))


def wavelet_scale_limits(
    L: int, P: int, jl: int, jp: int, lam_l: float, lam_p: float
) -> Tuple[int, int]:
    """computes the angular and radial band-limits for scale jl/jp"""
    return min(angular_bandlimit(jl, lam_l), L), min(
        radial_bandlimit(jp, lam_p), P
    )


def wavelet_scale_limits_N(
    L: int, P: int, N: int, jl: int, jp: int, lam_l: float, lam_p: float
) -> Tuple[int, int, int]:
    """computes the angular and radial band-limits and multiresolution directionality Nj
    for scale jl/jp
    """
    L_l = min(angular_bandlimit(jl, lam_l), L)
    L_p = min(radial_bandlimit(jp, lam_p), P)
    Nj = min(N, L_l)
    Nj += (Nj + N) % 2
    return L_l, L_p, Nj
