import numpy as np
import jax.numpy as jnp
from s2ball.transform import wigner
from s2ball.construct import matrix
from jax import jit
from functools import partial


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    P: int,
    tau: float = 1.0,
    matrices: np.ndarray = None,
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
        np.ndarray: Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    if matrices is None:
        matrices = matrix.generate_matrices(
            transform="wigner_laguerre",
            L=L,
            P=P,
            N=N,
            tau=tau,
            save_dir=save_dir,
        )

    shift = -1 if adjoint else 0

    if method == "numpy":
        return (
            inverse_transform(f, matrices, L, shift) * 2 * np.pi / (2 * N - 1)
            if adjoint
            else forward_transform(f, matrices, L, N)
        )
    elif method == "jax":
        return (
            inverse_transform_jax(f, matrices, L, shift) * 2 * jnp.pi / (2 * N - 1)
            if adjoint
            else forward_transform_jax(f, matrices, L, N)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(
    f: np.ndarray, matrices: np.ndarray, L: int, N: int, shift: int = 0
) -> np.ndarray:
    r"""Compute the forward Wigner-Laguerre transform with Numpy.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on the ball, with shape [P, 2N-1, L, 2L-1].
        matrices (np.ndarray): List of matrices corresponding to all
            necessary precomputed values.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        np.ndarray: Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmnr = wigner.forward_transform(f, matrices[0], L, N, shift)
    return np.einsum("...rnlm, rp -> ...pnlm", flmnr, matrices[1][shift])


@partial(jit, static_argnums=(2, 3, 4))
def forward_transform_jax(
    f: jnp.ndarray, matrices: jnp.ndarray, L: int, N: int, shift: int = 0
) -> jnp.ndarray:
    r"""Compute the forward Wigner-Laguerre transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        f (jnp.ndarray): Signal on the ball, with shape [P, 2N-1, L, 2L-1].
        matrices (np.ndarray): List of matrices corresponding to all
            necessary precomputed values.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        jnp.ndarray: Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmnr = wigner.forward_transform_jax(f, matrices[0], L, N, shift)
    return jnp.einsum(
        "...rnlm, rp -> ...pnlm", flmnr, matrices[1][shift], optimize=True
    )


def inverse(
    flmnp: np.ndarray,
    L: int,
    N: int,
    P: int,
    tau: float = 1.0,
    matrices: np.ndarray = None,
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
        np.ndarray: Signal on the ball, with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    if matrices is None:
        matrices = matrix.generate_matrices(
            transform="wigner_laguerre",
            L=L,
            P=P,
            N=N,
            tau=tau,
            save_dir=save_dir,
        )

    shift = -1 if adjoint else 0

    if method == "numpy":
        return (
            forward_transform(flmnp, matrices, L, N, shift) * (2 * N - 1) / (2 * np.pi)
            if adjoint
            else inverse_transform(flmnp, matrices, L)
        )
    elif method == "jax":
        return (
            forward_transform_jax(flmnp, matrices, L, N, shift)
            * (2 * N - 1)
            / (2 * jnp.pi)
            if adjoint
            else inverse_transform_jax(flmnp, matrices, L)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(
    flmnp: np.ndarray, matrices: np.ndarray, L: int, shift: int = 0
) -> np.ndarray:
    r"""Compute the inverse Wigner-Laguerre transform with Numpy.

    This transform trivially supports batching.

    Args:
        flmnp (np.ndarray): Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].
        matrices (np.ndarray): List of matrices corresponding to all
            necessary precomputed values.
        L (int): Harmonic band-limit.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        np.ndarray: Signal on the ball, with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmnr = np.einsum("...pnlm,rp -> ...rnlm", flmnp, matrices[1][1 + shift])
    return wigner.inverse_transform(flmnr, matrices[0], L, shift)


@partial(jit, static_argnums=(2, 3))
def inverse_transform_jax(
    flmnp: jnp.ndarray, matrices: jnp.ndarray, L: int, shift: int = 0
) -> jnp.ndarray:
    r"""Compute the inverse Wigner-Laguerre transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        flmnp (jnp.ndarray): Wigner-Laguerre coefficients with shape [P, 2N-1, L, 2L-1].
        matrices (np.ndarray): List of matrices corresponding to all
            necessary precomputed values.
        L (int): Harmonic band-limit.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        jnp.ndarray: Signal on the ball, with shape [P, 2N-1, L, 2L-1].

    Note:
        Currently only `Price & McEwen <https://arxiv.org/pdf/2105.05518.pdf>`_
        sampling on :math:`SO(3) \times \mathbb{R}^+` is supported, though this approach
        can be extended to alternate sampling schemes, e.g. HEALPix+.
    """
    flmnr = jnp.einsum(
        "...pnlm,rp -> ...rnlm", flmnp, matrices[1][1 + shift], optimize=True
    )
    return wigner.inverse_transform_jax(flmnr, matrices[0], L, shift)
