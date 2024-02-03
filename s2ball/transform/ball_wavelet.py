import numpy as np
import jax.numpy as jnp
from typing import Tuple, List
from s2ball.wavelets import tiling
from s2ball.construct import matrix
from s2ball.transform import laguerre, wigner_laguerre
from s2ball.wavelets.helper_functions import *
from jax import jit
from functools import partial


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    P: int,
    lam_l: float = 2.0,
    lam_p: float = 2.0,
    tau: float = 1.0,
    matrices: List[np.ndarray] = None,
    method: str = "jax",
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> Tuple[List[List[np.ndarray]], np.ndarray]:
    r"""Compute the forward directional wavelet transform on the ball.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        f (np.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float, optional): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets. Defaults to 2.0.
        lam_p (float, optional): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets. Defaults to 2.0.
        tau (float): Laguerre polynomial scale factor.
        matrices (List[np.ndarray], optional): List of matrices corresponding to all
            necessary precomputed values. Defaults to None.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        adjoint (bool, optional): Whether to return adjoint transformation.
            Defaults to False.

    Raises:
        ValueError: Method not in {"numpy", "jax"}.

    Returns:
        List[np.ndarray, List[List[np.ndarray]]]: Multiresolution scaling and wavelet coefficients


    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+. Also see `Price & McEwen
        <https://arxiv.org/pdf/2105.05518.pdf>`_ for details on directionality.
    """
    if matrices is None:
        matrices = matrix.generate_matrices(
            transform="wavelet",
            L=L,
            N=N,
            P=P,
            tau=tau,
            lam_l=lam_l,
            lam_p=lam_p,
            save_dir=save_dir,
        )

    shift = -1 if adjoint else 0

    if method == "numpy":
        return (
            inverse_transform(f, L, N, P, lam_l, lam_p, matrices, shift)
            if adjoint
            else forward_transform(f, L, N, P, lam_l, lam_p, matrices)
        )

    elif method == "jax":
        return (
            inverse_transform_jax(f, L, N, P, lam_l, lam_p, matrices, shift)
            if adjoint
            else forward_transform_jax(f, L, N, P, lam_l, lam_p, matrices)
        )

    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray,
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    matrices: List[np.ndarray],
    shift: int = 0,
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    r"""Compute the forward directional wavelet transform on the ball with Numpy.

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
        matrices (List[np.ndarray]): List of matrices corresponding to all
            necessary precomputed values.
        shift (int, optional): Shift for multiscale reindexing for adjoint transforms.

    Returns:
        List[np.ndarray, List[List[np.ndarray]]]: Multiresolution scaling and wavelet coefficients

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+. Also see `Price & McEwen
        <https://arxiv.org/pdf/2105.05518.pdf>`_ for details on directionality.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)
    flmp = laguerre.forward_transform(f, matrices[:2], shift)

    # Compute scaling coefficients
    f_scal_lmp = np.einsum("plm,pl->plm", flmp, matrices[4][shift][1])
    f_scal = laguerre.inverse_transform(f_scal_lmp, matrices[:2], shift)

    # Compute wavelet coefficients
    f_wav_lmnp = tiling.construct_f_wav_lmnp(L, N, P, lam_l, lam_p)

    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            f_wav_lmnp[jp][jl][:, ::2, :, :] = np.einsum(
                "plm,pln->pnlm",
                flmp[:L_p, :L_l, L - L_l : L - 1 + L_l],
                np.conj(matrices[4][1][0][jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2])
                * 2
                * np.pi
                / (2 * Nj - 1)
                if shift == -1
                else matrices[4][0][0][jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2],
            )
            f_wav_lmnp[jp][jl] = wigner_laguerre.inverse_transform(
                f_wav_lmnp[jp][jl], [matrices[2][jl], matrices[3][jp]], L_l, shift
            )

    return [f_scal, f_wav_lmnp]


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 7))
def forward_transform_jax(
    f: jnp.ndarray,
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    matrices: List[jnp.ndarray],
    shift: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, List[List[np.ndarray]]]:
    r"""Compute the forward directional wavelet transform on the ball with JAX and JIT.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        f (jnp.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        matrices (List[np.ndarray]): List of matrices corresponding to all
            necessary precomputed values.
        shift (int, optional): Shift for multiscale reindexing for adjoint transforms.

    Returns:
        List[jnp.ndarray, List[List[jnp.ndarray]]]: Multiresolution scaling and wavelet coefficients

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+. Also see `Price & McEwen
        <https://arxiv.org/pdf/2105.05518.pdf>`_ for details on directionality.
    """
    Jl = j_max(L, lam_l)
    Jp = j_max(P, lam_p)
    flmp = laguerre.forward_transform_jax(f, matrices[:2], shift)

    # Compute scaling coefficients
    f_scal_lmp = jnp.einsum("plm,pl->plm", flmp, matrices[4][shift][1], optimize=True)
    f_scal = laguerre.inverse_transform_jax(f_scal_lmp, matrices[:2], shift)

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
                        jnp.conj(
                            matrices[4][1][0][jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2]
                        )
                        * 2
                        * jnp.pi
                        / (2 * Nj - 1)
                        if shift == -1
                        else matrices[4][0][0][jp][jl][
                            :, :, L_l - Nj : L_l - 1 + Nj : 2
                        ],
                        optimize=True,
                    )
                )
            )
            f_wav_lmnp[jp][jl] = wigner_laguerre.inverse_transform_jax(
                f_wav_lmnp[jp][jl], [matrices[2][jl], matrices[3][jp]], L_l, shift
            )

    return [f_scal, f_wav_lmnp]


def inverse(
    w: Tuple[np.ndarray, List[List[np.ndarray]]],
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    tau: float = 1.0,
    matrices: List[np.ndarray] = None,
    method: str = "jax",
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> np.ndarray:
    r"""Compute the inverse directional wavelet transform on the ball.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        w (Tuple[np.ndarray, List[List[np.ndarray]]]): List containing scaling and wavelet
            coefficients in that order.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        tau (float): Laguerre polynomial scale factor.
        matrices (List[np.ndarray], optional): List of matrices corresponding to all
            necessary precomputed values. Defaults to None.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        adjoint (bool, optional): Whether to return adjoint transformation.
            Defaults to False.

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
    if matrices is None:
        matrices = matrix.generate_matrices(
            transform="wavelet",
            L=L,
            N=N,
            P=P,
            tau=tau,
            lam_l=lam_l,
            lam_p=lam_p,
            save_dir=save_dir,
        )

    shift = -1 if adjoint else 0

    if method == "numpy":
        return (
            forward_transform(w, L, N, P, lam_l, lam_p, matrices, shift)
            if adjoint
            else inverse_transform(w, L, N, P, lam_l, lam_p, matrices)
        )

    elif method == "jax":
        return (
            forward_transform_jax(w, L, N, P, lam_l, lam_p, matrices, shift)
            if adjoint
            else inverse_transform_jax(w, L, N, P, lam_l, lam_p, matrices)
        )

    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    w: Tuple[np.ndarray, List[List[np.ndarray]]],
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    matrices: List[np.ndarray],
    shift: int = 0,
) -> np.ndarray:
    r"""Compute the inverse directional wavelet transform on the ball with Numpy.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        w (Tuple[np.ndarray, List[List[np.ndarray]]]): List containing scaling and wavelet
            coefficients in that order.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        matrices (List[np.ndarray]): List of matrices corresponding to all
            necessary precomputed values.
        shift (int, optional): Shift for multiscale reindexing for adjoint transforms.

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
    f_scal_lmp = laguerre.forward_transform(w[0], matrices[:2], shift)
    flmp = np.einsum("plm, pl->plm", f_scal_lmp, matrices[4][1 + shift][1])

    # Compute wavelet coefficients
    f_wav_lmnp = tiling.construct_f_wav_lmnp(L, N, P, lam_l, lam_p)

    # Sum over the wavelet coefficients
    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            f_wav_lmnp[jp][jl] = wigner_laguerre.forward_transform(
                w[1][jp][jl], [matrices[2][jl], matrices[3][jp]], L_l, Nj, shift
            )
            flmp[:L_p, :L_l, L - L_l : L - 1 + L_l] += np.einsum(
                "pnlm, pln->plm",
                f_wav_lmnp[jp][jl][:, ::2, :, :],
                np.conj(matrices[4][0][0][jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2])
                * (2 * Nj - 1)
                / (2 * np.pi)
                if shift == -1
                else matrices[4][1][0][jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2],
            )

    return laguerre.inverse_transform(flmp, matrices[:2], shift)


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 7))
def inverse_transform_jax(
    w: Tuple[np.ndarray, List[List[np.ndarray]]],
    L: int,
    N: int,
    P: int,
    lam_l: float,
    lam_p: float,
    matrices: List[jnp.ndarray],
    shift: int = 0,
) -> jnp.ndarray:
    r"""Compute the inverse directional wavelet transform on the ball with JAX and JIT.

    This transform does not yet support batching, though this is straightforward
    to add.

    Args:
        w (Tuple[np.ndarray, List[List[np.ndarray]]]): List containing scaling and wavelet
            coefficients in that order.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        lam_l (float): Wavelet angular scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        lam_p (float): Wavelet radial scaling factor. :math:`\lambda = 2.0`
            indicates dyadic wavelets.
        matrices (List[np.ndarray]): List of matrices corresponding to all
            necessary precomputed values.
        shift (int, optional): Shift for multiscale reindexing for adjoint transforms.

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
    f_scal_lmp = laguerre.forward_transform_jax(w[0], matrices[:2], shift)
    flmp = jnp.einsum(
        "plm, pl->plm", f_scal_lmp, matrices[4][1 + shift][1], optimize=True
    )

    # Compute wavelet coefficients
    f_wav_lmnp = tiling.construct_f_wav_lmnp_jax(L, N, P, lam_l, lam_p)

    # Sum over the wavelet coefficients
    for jp in range(Jp + 1):
        for jl in range(Jl + 1):
            L_l, L_p, Nj = wavelet_scale_limits_N(L, P, N, jl, jp, lam_l, lam_p)
            f_wav_lmnp[jp][jl] = wigner_laguerre.forward_transform_jax(
                w[1][jp][jl], [matrices[2][jl], matrices[3][jp]], L_l, Nj, shift
            )
            flmp = flmp.at[:L_p, :L_l, L - L_l : L - 1 + L_l].add(
                jnp.einsum(
                    "pnlm, pln->plm",
                    f_wav_lmnp[jp][jl][:, ::2, :, :],
                    jnp.conj(
                        matrices[4][0][0][jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2]
                    )
                    * (2 * Nj - 1)
                    / (2 * jnp.pi)
                    if shift == -1
                    else matrices[4][1][0][jp][jl][:, :, L_l - Nj : L_l - 1 + Nj : 2],
                    optimize=True,
                )
            )

    return laguerre.inverse_transform_jax(flmp, matrices[:2], shift)
