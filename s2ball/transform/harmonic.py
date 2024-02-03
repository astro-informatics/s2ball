import numpy as np
import jax.numpy as jnp
from s2ball.construct import matrix
from jax import jit
from functools import partial


def forward(
    f: np.ndarray,
    L: int,
    legendre_matrices: np.ndarray = None,
    method: str = "jax",
    spin: int = 0,
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> np.ndarray:
    r"""Compute forward spherical harmonic transform.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on sphere, with shape: [L, 2L-1].
        L (int): Harmonic band-limit.
        legendre_matrices (np.ndarray, optional): List of Legendre transform matrices.
            Defaults to None.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        spin (int, optional): Harmonic spin. Defaults to 0.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        adjoint (bool, optional): Whether to return adjoint transformation.
            Defaults to False.

    Returns:
        np.ndarray: Spherical harmonic coefficients with shape [L, 2L-1].

    Raises:
        ValueError: Deployment method not in {"numpy", "jax"}.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    if legendre_matrices is None:
        matrices = matrix.generate_matrices(
            transform="spherical_harmonic", L=L, spin=spin, save_dir=save_dir
        )

    else:
        matrices = legendre_matrices

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
    f: np.ndarray, legendre_matrices: np.ndarray, shift: int = 0
) -> np.ndarray:
    r"""Compute the forward spherical harmonic transform with Numpy.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on sphere, with shape: [L, 2L-1].
        legendre_matrices (np.ndarray): List of Legendre transform matrices.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        np.ndarray: Spherical harmonic coefficients with shape [L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    fm = np.fft.fft(f)
    flm = np.einsum("...lmi, ...im->...lm", legendre_matrices[shift], fm)
    return np.fft.fftshift(flm, axes=-1)


@partial(jit, static_argnums=(2))
def forward_transform_jax(
    f: jnp.ndarray, legendre_matrices: jnp.ndarray, shift: int = 0
) -> jnp.ndarray:
    r"""Compute the forward spherical harmonic transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        f (jnp.ndarray): Signal on sphere, with shape: [L, 2L-1].
        legendre_matrices (jnp.ndarray): Legendre transform kernel.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        jnp.ndarray: Spherical harmonic coefficients with shape [L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    fm = jnp.fft.fft(f)
    flm = jnp.einsum(
        "...lmi, ...im->...lm", legendre_matrices[shift], fm, optimize=True
    )
    return jnp.fft.fftshift(flm, axes=-1)


def inverse(
    flm: np.ndarray,
    L: int,
    legendre_matrices: np.ndarray = None,
    method: str = "jax",
    spin: int = 0,
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> np.ndarray:
    r"""Compute the inverse spherical harmonic transform.

    This transform trivially supports batching.

    Args:
        flm (np.ndarray): Harmonic coefficients, with shape: [L, 2L-1].
        L (int): Harmonic band-limit.
        legendre_matrices (np.ndarray, optional): List of Legendre transform matrices.
            Defaults to None.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        spin (int, optional): Harmonic spin. Defaults to 0.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        adjoint (bool, optional): Whether to return adjoint transformation.
            Defaults to False.

    Returns:
        np.ndarray: Pixel-space coefficients with shape [L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    if legendre_matrices is None:
        matrices = matrix.generate_matrices(
            transform="spherical_harmonic", L=L, spin=spin, save_dir=save_dir
        )
    else:
        matrices = legendre_matrices

    shift = -1 if adjoint else 0

    if method == "numpy":
        return (
            forward_transform(flm, matrices, shift)
            if adjoint
            else inverse_transform(flm, matrices)
        )
    elif method == "jax":
        return (
            forward_transform_jax(flm, matrices, shift)
            if adjoint
            else inverse_transform_jax(flm, matrices)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(flm: np.ndarray, legendre_matrices: np.ndarray, shift: int = 0):
    r"""Compute the inverse spherical harmonic transform with Numpy.

    This transform trivially supports batching.

    Args:
        flm (np.ndarray): Harmonic coefficients, with shape: [L, 2L-1].
        legendre_matrices (np.ndarray): List of Legendre transform matrices.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        np.ndarray: Pixel-space coefficients with shape [L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    flm_shift = np.fft.ifftshift(flm, axes=-1)
    fm = np.einsum("...lmi, ...lm->...im", legendre_matrices[1 + shift], flm_shift)
    return np.fft.ifft(fm, norm="forward")


@partial(jit, static_argnums=(2))
def inverse_transform_jax(
    flm: jnp.ndarray, legendre_matrices: jnp.ndarray, shift: int = 0
) -> jnp.ndarray:
    r"""Compute the inverse spherical harmonic transform via with JAX and JIT.

    This transform trivially supports batching.

    Args:
        flm (jnp.ndarray): Harmonic coefficients, with shape: [L, 2L-1].
        legendre_matrices (jnp.ndarray): List of Legendre transform matrices.
        shift (int, optional): Used internally to handle adjoint transforms.

    Returns:
        jnp.ndarray: Pixel-space coefficients with shape [L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    flm_shift = jnp.fft.ifftshift(flm, axes=-1)
    fm = jnp.einsum(
        "...lmi, ...lm->...im", legendre_matrices[1 + shift], flm_shift, optimize=True
    )
    return jnp.fft.ifft(fm, norm="forward")
