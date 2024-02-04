import numpy as np
import jax.numpy as jnp
from s2ball.transform import harmonic
from s2ball.construct import matrix
from jax import jit
from functools import partial


def forward(
    f: np.ndarray,
    L: int,
    P: int,
    tau: float = 1.0,
    matrices: np.ndarray = None,
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
        tau (float, optional): Laguerre polynomial scale factor. Defaults to 1.0.
        matrices (np.ndarray, optional): List of matrices corresponding to all
            necessary precomputed values. Defaults to None.
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
    if matrices is None:
        matrices = matrix.generate_matrices(
            transform="spherical_laguerre",
            L=L,
            P=P,
            spin=spin,
            tau=tau,
            save_dir=save_dir,
        )

    shift = -1 if adjoint else 0

    if method == "numpy":
        return (
            inverse_transform(f, matrices, shift)
            if adjoint
            else forward_transform(f, matrices)
        )
    elif method == "jax":
        return (
            inverse_transform_jax(f, matrices, shift)
            if adjoint
            else forward_transform_jax(f, matrices)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray, matrices: np.ndarray, shift: int = 0
) -> np.ndarray:
    """Compute the forward spherical-Laguerre transform with Numpy.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        matrices (np.ndarray): List of matrices corresponding to all
            necessary precomputed values.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        np.ndarray: Spherical-Laguerre coefficients with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmr = harmonic.forward_transform(f, matrices[0], shift)
    return np.einsum("...rlm, rp -> ...plm", flmr, matrices[1][shift])


@partial(jit, static_argnums=(2))
def forward_transform_jax(
    f: jnp.ndarray, matrices: jnp.ndarray, shift: int = 0
) -> jnp.ndarray:
    """Compute the forward spherical-Laguerre transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        f (jnp.ndarray): Signal on the ball, with shape [P, L, 2L-1].
        matrices (jnp.ndarray): List of matrices corresponding to all
            necessary precomputed values.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        jnp.ndarray: Spherical-Laguerre coefficients with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmr = harmonic.forward_transform_jax(f, matrices[0], shift)
    return jnp.einsum("...rlm, rp -> ...plm", flmr, matrices[1][shift])


def inverse(
    flmp: np.ndarray,
    L: int,
    P: int,
    tau: float = 1.0,
    matrices: np.ndarray = None,
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
        tau (float, optional): Laguerre polynomial scale factor. Defaults to 1.0.
        matrices (np.ndarray, optional): List of matrices corresponding to all
            necessary precomputed values. Defaults to None.
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
    if matrices is None:
        matrices = matrix.generate_matrices(
            transform="spherical_laguerre",
            L=L,
            P=P,
            spin=spin,
            tau=tau,
            save_dir=save_dir,
        )

    shift = -1 if adjoint else 0

    if method == "numpy":
        return (
            forward_transform(flmp, matrices, shift)
            if adjoint
            else inverse_transform(flmp, matrices)
        )
    elif method == "jax":
        return (
            forward_transform_jax(flmp, matrices, shift)
            if adjoint
            else inverse_transform_jax(flmp, matrices)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    flmp: np.ndarray, matrices: np.ndarray, shift: int = 0
) -> np.ndarray:
    """Compute the inverse spherical-Laguerre transform with Numpy.

    This transform trivially supports batching.

    Args:
        flmp (np.ndarray): Spherical-Laguerre coefficients with shape [P, L, 2L-1].
        matrices (np.ndarray): List of matrices corresponding to all
            necessary precomputed values.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        np.ndarray: Signal on the ball, with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmr = np.einsum("...plm,rp -> ...rlm", flmp, matrices[1][1 + shift])
    return harmonic.inverse_transform(flmr, matrices[0], shift)


@partial(jit, static_argnums=(2))
def inverse_transform_jax(
    flmp: jnp.ndarray, matrices: jnp.ndarray, shift: int = 0
) -> jnp.ndarray:
    """Compute the inverse spherical-Laguerre transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        flmp (jnp.ndarray): Spherical-Laguerre coefficients with shape [P, L, 2L-1].
        matrices (jnp.ndarray): List of matrices corresponding to all
            necessary precomputed values.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        jnp.ndarray: Signal on the ball, with shape [P, L, 2L-1].

    Note:
        Currently only `Leistedt & McEwen <https://arxiv.org/pdf/1205.0792.pdf>`_
        sampling on the ball is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmr = jnp.einsum("...plm,rp -> ...rlm", flmp, matrices[1][1 + shift])
    return harmonic.inverse_transform_jax(flmr, matrices[0], shift)
