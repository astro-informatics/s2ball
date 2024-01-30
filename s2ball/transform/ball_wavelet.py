import numpy as np
import jax.numpy as jnp
from typing import Tuple, List
from s2ball.wavelets import tiling
from s2ball.construct import wavelet_constructor
from s2ball.transform import laguerre, wigner_laguerre
from s2ball.wavelets.helper_functions import *

from jax import jit
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    tau: float = 1.0,
    method: str = "jax",
    save_dir: str = ".matrices",
) -> Tuple[List[List[np.ndarray]], np.ndarray]:
    r"""Compute the forward directional wavelet transform on the ball.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        f (np.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        tau (float): Laguerre polynomial scale factor.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".

    Raises:
        ValueError: Method not in {"numpy", "jax"}.

    Returns:
        Tuple[List[List[np.ndarray]], np.ndarray]: Multiresolution wavelet coefficients
            and scaling coefficients.

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+. Also see `Price & McEwen
        <https://arxiv.org/pdf/2105.05518.pdf>`_ for details on directionality.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)

    wav_lmp, scal_lmp = tiling.compute_wav_lmp(L, N, P, lam_l, lam_p)
    scal_lmp *= np.sqrt((4 * np.pi) / (2 * np.arange(L) + 1))

    factor = 8 * np.pi**2 / (2 * np.arange(L) + 1)
    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l = angular_bandlimit(jl, lam_l)
            wav_lmp[jp][jl] = np.einsum(
                "pln,l->pln", np.conj(wav_lmp[jp][jl]), factor[:L_l]
            )

    # Precompute Legendre/Wigner transform kernels
    leg_kernels = wavelet_constructor.wavelet_legendre_kernels(L, save_dir)
    wig_kernels = wavelet_constructor.wavelet_wigner_kernels(
        L, N, lam_l, forward=False, save_dir=save_dir
    )
    # Precompute Laguerre polynomials for scaling/wavelet coefficients
    lag_polys = wavelet_constructor.scaling_laguerre_kernels(P, tau)
    wav_lag_polys = wavelet_constructor.wavelet_laguerre_kernels(
        P, lam_p, tau, forward=False
    )
    if method == "numpy":
        return forward_transform(
            f,
            wav_lmp,
            scal_lmp,
            L,
            N,
            P,
            lam_l,
            lam_p,
            leg_kernels,
            lag_polys,
            wig_kernels,
            wav_lag_polys,
        )

    elif method == "jax":
        return forward_transform_jax(
            f,
            wav_lmp,
            scal_lmp,
            L,
            N,
            P,
            lam_l,
            lam_p,
            leg_kernels,
            lag_polys,
            wig_kernels,
            wav_lag_polys,
        )

    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray,
    wav_lmp: List[List[np.ndarray]],
    scal_lmp: np.ndarray,
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    leg_kernels: Tuple[np.ndarray],
    lag_polys: Tuple[np.ndarray],
    wig_kernels: List[np.ndarray],
    wav_lag_polys: List[np.ndarray],
) -> np.ndarray:
    r"""Compute the forward directional wavelet transform on the ball with Numpy.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        f (np.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        wav_lmp (List[List[np.ndarray]]): Multiresolution wavelet filters.
        scal_lmp (np.ndarray): Scaling filters.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        leg_kernels (Tuple[np.ndarray]): Array of shape [2, L, 2L-1, L] which contains
            associated Legendre matrices.
        lag_polys (Tuple[np.ndarray]): Array of shape [2, R, P] which contains Laguerre
            polynomials sampled at radial nodes R.
        wig_kernels (List[np.ndarray]): List of Wigner transform kernels for each
            angular wavelet scale.
        wav_lag_polys (List[np.ndarray]): List of Laguerre polynomial kernels for each
            radial wavelet scale.

    Returns:
        Tuple[List[List[np.ndarray]], np.ndarray]: Multiresolution wavelet coefficients
            and scaling coefficients.

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+. Also see `Price & McEwen
        <https://arxiv.org/pdf/2105.05518.pdf>`_ for details on directionality.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)
    flmp = laguerre.forward_transform(f, leg_kernels[0], lag_polys[0])

    # Compute scaling coefficients
    f_scal_lmp = np.einsum("plm,pl->plm", flmp, scal_lmp)
    f_scal = laguerre.inverse_transform(f_scal_lmp, leg_kernels[1], lag_polys[1])

    # Compute wavelet coefficients
    f_wav_lmnp = tiling.construct_f_wav_lmnp(L, N, P, lam_l, lam_p)

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            f_wav_lmnp[jp][jl][:, ::2, :, :] = np.einsum(
                "plm,pln->pnlm",
                flmp[:L_p, :L_l, L - L_l : L - 1 + L_l],
                wav_lmp[jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2],
            )
            f_wav_lmnp[jp][jl] = wigner_laguerre.inverse_transform(
                f_wav_lmnp[jp][jl], wig_kernels[jl], wav_lag_polys[jp], L_l
            )

    return f_wav_lmnp, f_scal


@partial(jit, static_argnums=(3, 4, 5, 6, 7))
def forward_transform_jax(
    f: jnp.ndarray,
    wav_lmp: List[List[jnp.ndarray]],
    scal_lmp: jnp.ndarray,
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    leg_kernels: Tuple[jnp.ndarray],
    lag_polys: Tuple[jnp.ndarray],
    wig_kernels: List[jnp.ndarray],
    wav_lag_polys: List[jnp.ndarray],
) -> jnp.ndarray:
    r"""Compute the forward directional wavelet transform on the ball with JAX and JIT.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        f (jnp.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        wav_lmp (List[List[jnp.ndarray]]): Multiresolution wavelet filters.
        scal_lmp (jnp.ndarray): Scaling filters.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        leg_kernels (Tuple[jnp.ndarray]): Array of shape [2, L, 2L-1, L] which contains
            associated Legendre matrices.
        lag_polys (Tuple[jnp.ndarray]): Array of shape [2, R, P] which contains Laguerre
            polynomials sampled at radial nodes R.
        wig_kernels (List[jnp.ndarray]): List of Wigner transform kernels for each
            angular wavelet scale.
        wav_lag_polys (List[jnp.ndarray]): List of Laguerre polynomial kernels for each
            radial wavelet scale.

    Returns:
        Tuple[List[List[jnp.ndarray]], jnp.ndarray]: Multiresolution wavelet coefficients
            and scaling coefficients.

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+. Also see `Price & McEwen
        <https://arxiv.org/pdf/2105.05518.pdf>`_ for details on directionality.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)
    flmp = laguerre.forward_transform_jax(f, leg_kernels[0], lag_polys[0])

    # Compute scaling coefficients
    f_scal_lmp = jnp.einsum("plm,pl->plm", flmp, scal_lmp, optimize=True)
    f_scal = laguerre.inverse_transform_jax(f_scal_lmp, leg_kernels[1], lag_polys[1])

    # Compute wavelet coefficients
    f_wav_lmnp = tiling.construct_f_wav_lmnp_jax(L, N, P, lam_l, lam_p)

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            f_wav_lmnp[jp][jl] = (
                f_wav_lmnp[jp][jl]
                .at[:, ::2, :, :]
                .set(
                    jnp.einsum(
                        "plm,pln->pnlm",
                        flmp[:L_p, :L_l, L - L_l : L - 1 + L_l],
                        wav_lmp[jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2],
                        optimize=True,
                    )
                )
            )
            f_wav_lmnp[jp][jl] = wigner_laguerre.inverse_transform_jax(
                f_wav_lmnp[jp][jl], wig_kernels[jl], wav_lag_polys[jp], L_l
            )

    return f_wav_lmnp, f_scal


def inverse(
    f_wav: List[List[np.ndarray]],
    f_scal: np.ndarray,
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    tau: float = 1.0,
    method: str = "jax",
    save_dir: str = ".matrices",
) -> np.ndarray:
    r"""Compute the inverse directional wavelet transform on the ball.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        f_wav (List[List[np.ndarray]]): Multiresolution wavelet coefficients.
        f_scal (np.ndarray): Scaling coefficients.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        tau (float): Laguerre polynomial scale factor.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".

    Raises:
        ValueError: Method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: Signal on the ball, with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+. Also see `Price & McEwen
        <https://arxiv.org/pdf/2105.05518.pdf>`_ for details on directionality.
    """
    wav_lmp, scal_lmp = tiling.compute_wav_lmp(L, N, P, lam_l, lam_p)
    scal_lmp *= np.sqrt((4 * np.pi) / (2 * np.arange(L) + 1))

    # Precompute Legendre/Wigner transform kernels
    leg_kernels = wavelet_constructor.wavelet_legendre_kernels(L, save_dir)
    wig_kernels = wavelet_constructor.wavelet_wigner_kernels(
        L, N, lam_l, forward=True, save_dir=save_dir
    )
    # Precompute Laguerre polynomials for scaling/wavelet coefficients
    lag_polys = wavelet_constructor.scaling_laguerre_kernels(P, tau)
    wav_lag_polys = wavelet_constructor.wavelet_laguerre_kernels(
        P, lam_p, tau, forward=True
    )
    if method == "numpy":

        return inverse_transform(
            f_wav,
            f_scal,
            wav_lmp,
            scal_lmp,
            L,
            N,
            P,
            lam_l,
            lam_p,
            leg_kernels,
            lag_polys,
            wig_kernels,
            wav_lag_polys,
        )

    elif method == "jax":
        return inverse_transform_jax(
            f_wav,
            f_scal,
            wav_lmp,
            scal_lmp,
            L,
            N,
            P,
            lam_l,
            lam_p,
            leg_kernels,
            lag_polys,
            wig_kernels,
            wav_lag_polys,
        )

    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    f_wav: List[List[np.ndarray]],
    f_scal: np.ndarray,
    wav_lmp: List[List[np.ndarray]],
    scal_lmp: np.ndarray,
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    leg_kernels: Tuple[np.ndarray],
    lag_polys: Tuple[np.ndarray],
    wig_kernels: List[np.ndarray],
    wav_lag_polys: List[np.ndarray],
) -> np.ndarray:
    r"""Compute the inverse directional wavelet transform on the ball with Numpy.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        f_wav (List[List[np.ndarray]]): Multiresolution wavelet coefficients.
        f_scal (np.ndarray): Scaling coefficients.
        wav_lmp (List[List[np.ndarray]]): Multiresolution wavelet filters.
        scal_lmp (np.ndarray): Scaling filters.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        leg_kernels (Tuple[np.ndarray]): Array of shape [2, L, 2L-1, L] which contains
            associated Legendre matrices.
        lag_polys (Tuple[np.ndarray]): Array of shape [2, R, P] which contains Laguerre
            polynomials sampled at radial nodes R.
        wig_kernels (List[np.ndarray]): List of Wigner transform kernels for each
            angular wavelet scale.
        wav_lag_polys (List[np.ndarray]): List of Laguerre polynomial kernels for each
            radial wavelet scale.

    Returns:
        np.ndarray: Signal on the ball, with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+. Also see `Price & McEwen
        <https://arxiv.org/pdf/2105.05518.pdf>`_ for details on directionality.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)

    # Sum over the scaling coefficients
    f_scal_lmp = laguerre.forward_transform(f_scal, leg_kernels[0], lag_polys[0])
    flmp = np.einsum("plm, pl->plm", f_scal_lmp, scal_lmp)

    # Compute wavelet coefficients
    f_wav_lmnp = tiling.construct_f_wav_lmnp(L, N, P, lam_l, lam_p)

    # Sum over the wavelet coefficients
    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            f_wav_lmnp[jp][jl] = wigner_laguerre.forward_transform(
                f_wav[jp][jl], wig_kernels[jl], wav_lag_polys[jp], L_l, Nj
            )
            flmp[:L_p, :L_l, L - L_l : L - 1 + L_l] += np.einsum(
                "pnlm, pln->plm",
                f_wav_lmnp[jp][jl][:, ::2, :, :],
                wav_lmp[jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2],
            )

    return laguerre.inverse_transform(flmp, leg_kernels[1], lag_polys[1])


@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def inverse_transform_jax(
    f_wav: List[List[jnp.ndarray]],
    f_scal: jnp.ndarray,
    wav_lmp: List[List[jnp.ndarray]],
    scal_lmp: jnp.ndarray,
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    leg_kernels: Tuple[jnp.ndarray],
    lag_polys: Tuple[jnp.ndarray],
    wig_kernels: List[jnp.ndarray],
    wav_lag_polys: List[jnp.ndarray],
) -> jnp.ndarray:
    r"""Compute the inverse directional wavelet transform on the ball with JAX and JIT.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        f_wav (List[List[jnp.ndarray]]): Multiresolution wavelet coefficients.
        f_scal (jnp.ndarray): Scaling coefficients.
        wav_lmp (List[List[jnp.ndarray]]): Multiresolution wavelet filters.
        scal_lmp (jnp.ndarray): Scaling filters.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        leg_kernels (Tuple[jnp.ndarray]): Array of shape [2, L, 2L-1, L] which contains
            associated Legendre matrices.
        lag_polys (Tuple[jnp.ndarray]): Array of shape [2, R, P] which contains Laguerre
            polynomials sampled at radial nodes R.
        wig_kernels (List[jnp.ndarray]): List of Wigner transform kernels for each
            angular wavelet scale.
        wav_lag_polys (List[jnp.ndarray]): List of Laguerre polynomial kernels for each
            radial wavelet scale.

    Returns:
        jnp.ndarray: Signal on the ball, with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+. Also see `Price & McEwen
        <https://arxiv.org/pdf/2105.05518.pdf>`_ for details on directionality.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)
    # Sum over the scaling coefficients
    f_scal_lmp = laguerre.forward_transform_jax(f_scal, leg_kernels[0], lag_polys[0])
    flmp = jnp.einsum("plm, pl->plm", f_scal_lmp, scal_lmp, optimize=True)

    # Compute wavelet coefficients
    f_wav_lmnp = tiling.construct_f_wav_lmnp_jax(L, N, P, lam_l, lam_p)

    # Sum over the wavelet coefficients
    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            f_wav_lmnp[jp][jl] = wigner_laguerre.forward_transform_jax(
                f_wav[jp][jl], wig_kernels[jl], wav_lag_polys[jp], L_l, Nj
            )
            flmp = flmp.at[:L_p, :L_l, L - L_l : L - 1 + L_l].add(
                jnp.einsum(
                    "pnlm, pln->plm",
                    f_wav_lmnp[jp][jl][:, ::2, :, :],
                    wav_lmp[jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2],
                    optimize=True,
                )
            )

    return laguerre.inverse_transform_jax(flmp, leg_kernels[1], lag_polys[1])
