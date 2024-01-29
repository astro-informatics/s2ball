import numpy as np
import jax.numpy as jnp
from baller.transform import wigner
from baller.sampling import laguerre_sampling
from baller.construct import wigner_constructor

from jax import jit
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    P: int,
    tau: float,
    kernel: np.ndarray = None,
    lag_poly: np.ndarray = None,
    method: str = "jax",
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> np.ndarray:
    r"""Compute the forward Wigner-Laguerre transform.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on the ball, with shape [P, 2N-1, L, 2L-1].
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        tau (float): Laguerre polynomial scale factor.
        kernel (np.ndarray, optional): Legendre transform kernel. Defaults to None.
        lag_poly (np.ndarray, optional): Laguerre polynomial kernel. Defaults to None.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        spin (int, optional): _description_. Defaults to 0.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        adjoint (bool, optional): Whether to return adjoint transformation.
            Defaults to False.

    Raises:
        ValueError: Method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    if lag_poly is None:
        lag_poly = laguerre_sampling.polynomials(P, tau, forward=True)
    if kernel is None:
        kernel = wigner_constructor.load_wigner_matrix(
            L=L, N=N, forward=True, save_dir=save_dir
        )
    if method == "numpy":
        return (
            inverse_transform(f, kernel, lag_poly, L) * 2 * np.pi / (2 * N - 1)
            if adjoint
            else forward_transform(f, kernel, lag_poly, L, N)
        )
    elif method == "jax":
        return (
            inverse_transform_jax(f, kernel, lag_poly, L)
            * 2
            * jnp.pi
            / (2 * N - 1)
            if adjoint
            else forward_transform_jax(f, kernel, lag_poly, L, N)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray, kernel: np.ndarray, lag_poly: np.ndarray, L: int, N: int
) -> np.ndarray:
    r"""Compute the forward Wigner-Laguerre transform with Numpy.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on the ball, with shape [P, 2N-1, L, 2L-1].
        kernel (np.ndarray, optional): Legendre transform kernel. Defaults to None.
        lag_poly (np.ndarray, optional): Laguerre polynomial kernel. Defaults to None.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.

    Returns:
        np.ndarray: Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmnr = wigner.forward_transform(f, kernel, L, N)
    return np.einsum("...rnlm, rp -> ...pnlm", flmnr, lag_poly)


@partial(jit, static_argnums=(3, 4))
def forward_transform_jax(
    f: jnp.ndarray, kernel: jnp.ndarray, lag_poly: jnp.ndarray, L: int, N: int
) -> jnp.ndarray:
    r"""Compute the forward Wigner-Laguerre transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        f (jnp.ndarray): Signal on the ball, with shape [P, 2N-1, L, 2L-1].
        kernel (jnp.ndarray, optional): Legendre transform kernel. Defaults to None.
        lag_poly (jnp.ndarray, optional): Laguerre polynomial kernel. Defaults to None.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.

    Returns:
        jnp.ndarray: Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmnr = wigner.forward_transform_jax(f, kernel, L, N)
    return jnp.einsum("...rnlm, rp -> ...pnlm", flmnr, lag_poly, optimize=True)


def inverse(
    flmnp: np.ndarray,
    L: int,
    N: int,
    P: int,
    tau: float,
    kernel: np.ndarray = None,
    lag_poly: np.ndarray = None,
    method: str = "jax",
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> np.ndarray:
    r"""Compute the inverse Wigner-Laguerre transform.

    This transform trivially supports batching.

    Args:
        flmnp (np.ndarray): Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.
        tau (float): Laguerre polynomial scale factor.
        kernel (np.ndarray, optional): Legendre transform kernel. Defaults to None.
        lag_poly (np.ndarray, optional): Laguerre polynomial kernel. Defaults to None.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        spin (int, optional): _description_. Defaults to 0.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        adjoint (bool, optional): Whether to return adjoint transformation.
            Defaults to False.

    Raises:
        ValueError: Method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: Signal on the ball, with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    if lag_poly is None:
        lag_poly = laguerre_sampling.polynomials(P, tau, forward=False)
    if kernel is None:
        kernel = wigner_constructor.load_wigner_matrix(
            L=L, N=N, forward=False, save_dir=save_dir
        )

    if method == "numpy":
        return (
            forward_transform(flmnp, kernel, lag_poly, L, N)
            * (2 * N - 1)
            / (2 * np.pi)
            if adjoint
            else inverse_transform(flmnp, kernel, lag_poly, L)
        )
    elif method == "jax":
        return (
            forward_transform_jax(flmnp, kernel, lag_poly, L, N)
            * (2 * N - 1)
            / (2 * jnp.pi)
            if adjoint
            else inverse_transform_jax(flmnp, kernel, lag_poly, L)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    flmnp: np.ndarray, kernel: np.ndarray, lag_poly: np.ndarray, L: int
) -> np.ndarray:
    r"""Compute the inverse Wigner-Laguerre transform with Numpy.

    This transform trivially supports batching.

    Args:
        flmnp (np.ndarray): Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].
        kernel (np.ndarray, optional): Legendre transform kernel. Defaults to None.
        lag_poly (np.ndarray, optional): Laguerre polynomial kernel. Defaults to None.
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Signal on the ball, with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmnr = np.einsum("...pnlm,rp -> ...rnlm", flmnp, lag_poly)
    return wigner.inverse_transform(flmnr, kernel, L)


@partial(jit, static_argnums=(3))
def inverse_transform_jax(
    flmnp: jnp.ndarray,
    kernel: jnp.ndarray,
    lag_poly: jnp.ndarray,
    L: int,
) -> jnp.ndarray:
    r"""Compute the inverse Wigner-Laguerre transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        flmnp (jnp.ndarray): Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].
        kernel (jnp.ndarray, optional): Legendre transform kernel. Defaults to None.
        lag_poly (jnp.ndarray, optional): Laguerre polynomial kernel. Defaults to None.
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Signal on the ball, with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmnr = jnp.einsum("...pnlm,rp -> ...rnlm", flmnp, lag_poly, optimize=True)
    return wigner.inverse_transform_jax(flmnr, kernel, L)
