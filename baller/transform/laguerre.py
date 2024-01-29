import numpy as np
import jax.numpy as jnp
from baller.transform import harmonic
from baller.sampling import laguerre_sampling
from baller.construct import legendre_constructor

from jax import jit
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)


def forward(
    f: np.ndarray,
    L: int,
    P: int,
    tau: float,
    kernel: np.ndarray = None,
    lag_poly: np.ndarray = None,
    method: str = "jax",
    spin: int = 0,
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> np.ndarray:
    """Compute the forward spherical-Laguerre transform.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        L (int): Harmonic band-limit.
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
        ValueError: Deployment method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: Spherical-Laguerre coefficients with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """
    if lag_poly is None:
        lag_poly = laguerre_sampling.polynomials(P, tau, forward=True)
    if kernel is None:
        kernel = legendre_constructor.load_legendre_matrix(
            L=L, forward=True, spin=spin, save_dir=save_dir
        )
    if method == "numpy":
        return (
            inverse_transform(f, kernel, lag_poly)
            if adjoint
            else forward_transform(f, kernel, lag_poly)
        )
    elif method == "jax":
        return (
            inverse_transform_jax(f, kernel, lag_poly)
            if adjoint
            else forward_transform_jax(f, kernel, lag_poly)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray,
    kernel: np.ndarray,
    lag_poly: np.ndarray,
) -> np.ndarray:
    """Compute the forward spherical-Laguerre transform with Numpy.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        kernel (np.ndarray): Legendre transform kernel.
        lag_poly (np.ndarray): Laguerre polynomial kernel.

    Returns:
        np.ndarray: Spherical-Laguerre coefficients with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmr = harmonic.forward_transform(f, kernel)
    return np.einsum("...rlm, rp -> ...plm", flmr, lag_poly)


@partial(jit)
def forward_transform_jax(
    f: jnp.ndarray,
    kernel: jnp.ndarray,
    lag_poly: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the forward spherical-Laguerre transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        f (jnp.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        kernel (jnp.ndarray): Legendre transform kernel.
        lag_poly (jnp.ndarray): Laguerre polynomial kernel.

    Returns:
        jnp.ndarray: Spherical-Laguerre coefficients with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmr = harmonic.forward_transform_jax(f, kernel)
    return jnp.einsum("...rlm, rp -> ...plm", flmr, lag_poly)


def inverse(
    flmp: np.ndarray,
    L: int,
    P: int,
    tau: float,
    kernel: np.ndarray = None,
    lag_poly: np.ndarray = None,
    method: str = "jax",
    spin: int = 0,
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> np.ndarray:
    """Compute the inverse spherical-Laguerre transform.

    This transform trivially supports batching.

    Args:
        flmp (np.ndarray): Spherical-Laguerre coefficients with shape [P, L, 2L-1].
        L (int): Harmonic band-limit.
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
        np.ndarray: Signal on the ball, with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """

    if lag_poly is None:
        lag_poly = laguerre_sampling.polynomials(P, tau, forward=False)
    if kernel is None:
        kernel = legendre_constructor.load_legendre_matrix(
            L=L, forward=False, spin=spin, save_dir=save_dir
        )

    if method == "numpy":
        return (
            forward_transform(flmp, kernel, lag_poly)
            if adjoint
            else inverse_transform(flmp, kernel, lag_poly)
        )
    elif method == "jax":
        return (
            forward_transform_jax(flmp, kernel, lag_poly)
            if adjoint
            else inverse_transform_jax(flmp, kernel, lag_poly)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    flmp: np.ndarray,
    kernel: np.ndarray,
    lag_poly: np.ndarray,
) -> np.ndarray:
    """Compute the inverse spherical-Laguerre transform with Numpy.

    This transform trivially supports batching.

    Args:
        flmp (np.ndarray): Spherical-Laguerre coefficients with shape [P, L, 2L-1].
        kernel (np.ndarray): Legendre transform kernel. Defaults to None.
        lag_poly (np.ndarray): Laguerre polynomial kernel. Defaults to None.

    Returns:
        np.ndarray: Signal on the ball, with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmr = np.einsum("...plm,rp -> ...rlm", flmp, lag_poly)
    return harmonic.inverse_transform(flmr, kernel)


@partial(jit)
def inverse_transform_jax(
    flmp: jnp.ndarray,
    kernel: jnp.ndarray,
    lag_poly: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the inverse spherical-Laguerre transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        flmp (jnp.ndarray): Spherical-Laguerre coefficients with shape [P, L, 2L-1].
        kernel (jnp.ndarray): Legendre transform kernel. Defaults to None.
        lag_poly (jnp.ndarray): Laguerre polynomial kernel. Defaults to None.

    Returns:
        jnp.ndarray: Signal on the ball, with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmr = jnp.einsum("...plm,rp -> ...rlm", flmp, lag_poly)
    return harmonic.inverse_transform_jax(flmr, kernel)
