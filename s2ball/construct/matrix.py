import numpy as np
from s2ball.sampling import laguerre_sampling
from s2ball.construct import (
    legendre_constructor,
    wigner_constructor,
    wavelet_constructor,
)
from s2ball.wavelets import helper_functions, tiling


def generate_matrices(
    transform: str,
    L: int,
    N: int = None,
    P: int = None,
    spin: int = 0,
    tau: float = 1.0,
    lam_l: float = 2.0,
    lam_p: float = 2.0,
    save_dir: str = ".matrices",
):
    """Top level wrapper function which handles all precompute matrices.

    Args:
        transform (str): Name of transform to compute/laod matrices for.
            Choice from ["spherical_harmonic, wigner, spherical_laguerre,
            wigner_laguerre, wavelet"].
        L (int): Harmonic band-limit.
        N (int, optional): Azimuthal band-limit. Defaults to None.
        P (int, optional): Radial band-limit. Defaults to None.
        spin (int, optional): Spin number of field. Defaults to 0.
        tau (float, optional): Laguerre polynomial scale factor. Defaults to 1.0.
        lam_l (float, optional): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets. Defaults to 2.0.
        lam_p (float, optional): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets. Defaults to 2.0.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".

    Notes:
        We enforce the following ordering convention: where appropriate order by
        - Legendre
        - Laguerre
        - Wigner
        - Wavelet Filer
        with each entry having forward/inverse as first and second element resp.

    Returns:
        List[np.ndarray]: Collection of np.ndarrays which contain computed matrice elements.
    """
    # Check transform exists
    if transform.lower() not in [
        "spherical_harmonic",
        "wigner",
        "spherical_laguerre",
        "wigner_laguerre",
        "wavelet",
    ]:
        raise ValueError("Specified transform does not exist.")

    # Check whether parameters and transform string match
    if transform.lower() == "spherical_harmonic":
        return __legendre(L, spin, save_dir)

    if transform.lower() == "wigner":
        return __wigner(L, N, save_dir)

    if transform.lower() == "spherical_laguerre":
        return [__legendre(L, spin, save_dir), __laguerre(P, tau)]

    if transform.lower() == "wigner_laguerre":
        return [__wigner(L, N, save_dir), __laguerre(P, tau)]

    if transform.lower() == "wavelet":
        legendre = __legendre(L, spin, save_dir)
        laguerre = __laguerre(P, tau)
        multi_wigner = __multiscale_wigner(L, N, lam_l, save_dir)
        multi_laguerre = __multiscale_laguerre(P, tau, lam_p)
        filters = __filters(L, N, P, lam_l, lam_p)
        return [legendre, laguerre, multi_wigner, multi_laguerre, filters]


def __legendre(L: int, spin: int, save_dir: str):
    """Private wrapper for Legendre matrix constructor switch"""
    legendre = []
    legendre.append(legendre_constructor.load_legendre_matrix(L, save_dir, True, spin))
    legendre.append(legendre_constructor.load_legendre_matrix(L, save_dir, False, spin))
    return legendre


def __wigner(L: int, N: int, save_dir: str):
    """Private wrapper for Wigner matrix constructor switch"""
    if N == None:
        raise ValueError("Azimuthal bandlimit N not provided for Wigner transform.")
    wigner = []
    wigner.append(wigner_constructor.load_wigner_matrix(L, N, save_dir, True))
    wigner.append(wigner_constructor.load_wigner_matrix(L, N, save_dir, False))
    return wigner


def __laguerre(P: int, tau: float):
    """Private wrapper for Laguerre polynomial matrix constructor switch"""
    if P == None:
        raise ValueError("Radial bandlimit P not provided for Laguerre transform.")
    laguerre = []
    laguerre.append(laguerre_sampling.polynomials(P, tau, True))
    laguerre.append(laguerre_sampling.polynomials(P, tau, False))
    return laguerre


def __multiscale_wigner(L: int, N: int, lam_l: float, save_dir: str):
    """Private wrapper for multiscale Wigner matrix constructor switch"""
    if N == None:
        raise ValueError("Azimuthal bandlimit N not provided for Wigner transform.")
    return wavelet_constructor.wavelet_wigner_kernels(L, N, lam_l, save_dir)


def __multiscale_laguerre(P: int, tau: float, lam_p: float):
    """Private wrapper for multiscale Laguerre matrix constructor switch"""
    if P == None:
        raise ValueError("Radial bandlimit P not provided for Laguerre transform.")
    return wavelet_constructor.wavelet_laguerre_kernels(P, lam_p, tau)


def __filters(L: int, N: int, P: int, lam_l: float, lam_p: float):
    """Private wrapper for wavelet filter constructor switch"""
    if N == None:
        raise ValueError("Azimuthal bandlimit N not provided for Wigner transform.")
    elif P == None:
        raise ValueError("Radial bandlimit P not provided for Laguerre transform.")
    Jl = helper_functions.j_max(L, lam_l)
    Jp = helper_functions.j_max(P, lam_p)
    wav_lmp_i, scal_lmp = tiling.compute_wav_lmp(L, N, P, lam_l, lam_p)
    scal_lmp *= np.sqrt((4 * np.pi) / (2 * np.arange(L) + 1))

    wav_lmp_f = tiling.construct_wav_lmp(L, P, lam_l, lam_p)
    factor = 8 * np.pi**2 / (2 * np.arange(L) + 1)
    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l = helper_functions.angular_bandlimit(jl, lam_l)
            wav_lmp_f[jp][jl] = np.einsum(
                "pln,l->pln", np.conj(wav_lmp_i[jp][jl]), factor[:L_l]
            )
    return [[wav_lmp_f, scal_lmp], [wav_lmp_i, scal_lmp]]
