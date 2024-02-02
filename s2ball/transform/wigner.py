import numpy as np
import jax.numpy as jnp
import s2ball

from functools import partial
from jax import jit, device_put
from jax.config import config

config.update("jax_enable_x64", True)


def forward(
    f: np.ndarray,
    L: int,
    N: int,
    wigner_kernel: np.ndarray = None,
    method: str = "jax",
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> np.ndarray:
    r"""Compute forward Wigner transform.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on rotation group, with shape: [2N-1, L, 2L-1].
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        wigner_kernel (np.ndarray): Wigner transform kernel.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        adjoint (bool, optional): Whether to return adjoint transformation.
            Defaults to False.

    Raises:
        ValueError: Method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: Wigner coefficients with shape [2N-1, L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    if wigner_kernel is None:
        kernel = s2ball.construct.wigner_constructor.load_wigner_matrix(
            L=L, N=N, forward=True, save_dir=save_dir
        )
    else:
        kernel = wigner_kernel

    if method == "numpy":
        return (
            inverse_transform(f, kernel, L) * 2 * np.pi / (2 * N - 1)
            if adjoint
            else forward_transform(f, kernel, L, N)
        )
    elif method == "jax":
        return (
            inverse_transform_jax(f, kernel, L) * 2 * jnp.pi / (2 * N - 1)
            if adjoint
            else forward_transform_jax(f, kernel, L, N)
        )
    else:
        raise ValueError(f"Method {method} not recognised.")


def forward_transform(f: np.ndarray, kernel: np.ndarray, L: int, N: int) -> np.ndarray:
    r"""Compute the forward Wigner transform with Numpy.

    This transform trivially supports batching.

    Args:
        f (np.ndarray): Signal on rotation group, with shape: [2N-1, L, 2L-1].
        kernel (np.ndarray): Wigner transform kernel.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.

    Returns:
        np.ndarray: Wigner coefficients with shape [2N-1, L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    fnab = np.fft.fftshift(np.fft.fft(f, axis=-3), axes=-3)
    fnab *= 2 * np.pi / (2 * N - 1)
    fnam = np.fft.fft(fnab, axis=-1)
    fnlm = np.einsum("...lmi, ...im->...lm", kernel, fnam)
    return np.fft.fftshift(fnlm, axes=-1)


@partial(jit, static_argnums=(2, 3))
def forward_transform_jax(
    f: jnp.ndarray, kernel: jnp.ndarray, L: int, N: int
) -> jnp.ndarray:
    r"""Compute the forward Wigner transform with JAX and JIT.

    This transform trivially supports batching.

    Args:
        f (jnp.ndarray): Signal on rotation group, with shape: [2N-1, L, 2L-1].
        kernel (jnp.ndarray): Wigner transform kernel.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.

    Returns:
        jnp.ndarray: Wigner coefficients with shape [2N-1, L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    fnab = jnp.fft.fftshift(jnp.fft.fft(f, axis=-3), axes=-3)
    fnab *= 2 * jnp.pi / (2 * N - 1)
    fnam = jnp.fft.fft(fnab, axis=-1)
    fnlm = jnp.einsum("...lmi, ...im->...lm", kernel, fnam, optimize=True)
    return jnp.fft.fftshift(fnlm, axes=-1)


def inverse(
    fnlm: np.ndarray,
    L: int,
    N: int,
    wigner_kernel: np.ndarray = None,
    method: str = "jax",
    save_dir: str = ".matrices",
    adjoint: bool = False,
) -> np.ndarray:
    r"""Compute the inverse Wigner transform.

    This transform trivially supports batching.

    Args:
        fnlm (np.ndarray): Wigner coefficients, with shape: [2N-1, L, 2L-1].
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        wigner_kernel (np.ndarray): Wigner transform kernel.
        method (str, optional): Evaluation method in {"numpy", "jax"}.
            Defaults to "jax".
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        adjoint (bool, optional): Whether to return adjoint transformation.
            Defaults to False.

    Raises:
        ValueError: Deployment method not in {"numpy", "jax"}.

    Returns:
        np.ndarray: Pixel-space coefficients with shape [2N-1, L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    if wigner_kernel is None:
        kernel = s2ball.construct.wigner_constructor.load_wigner_matrix(
            L=L, N=N, forward=False, save_dir=save_dir
        )
    else:
        kernel = wigner_kernel

    if method == "numpy":
        return (
            forward_transform(fnlm, kernel, L, N) * (2 * N - 1) / (2 * np.pi)
            if adjoint
            else inverse_transform(fnlm, kernel, L)
        )
    elif method == "jax":
        return (
            forward_transform_jax(fnlm, kernel, L, N) * (2 * N - 1) / (2 * jnp.pi)
            if adjoint
            else inverse_transform_jax(fnlm, kernel, L)
        )

    else:
        raise ValueError(f"Method {method} not recognised.")


def inverse_transform(fnlm: np.ndarray, kernel: np.ndarray, L: int) -> np.ndarray:
    r"""Compute the inverse Wigner transform with Numpy.

    Args:
        fnlm (np.ndarray): Wigner coefficients, with shape: [2N-1, L, 2L-1].
        kernel (np.ndarray): Wigner transform kernel.
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: Pixel-space coefficients with shape [2N-1, L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    fnlm_shift = np.fft.ifftshift(fnlm, axes=-1)
    fnam = np.einsum("...lmi, ...lm->...im", kernel, fnlm_shift)
    fnab = np.fft.ifft(fnam, axis=-1, norm="forward")
    return np.fft.ifft(np.fft.ifftshift(fnab, axes=-3), norm="forward", axis=-3)


@partial(jit, static_argnums=(2))
def inverse_transform_jax(
    fnlm: jnp.ndarray, kernel: jnp.ndarray, L: int
) -> jnp.ndarray:
    r"""Compute the inverse Wigner transform via precompute (JAX implementation).

    Args:
        fnlm (jnp.ndarray): Wigner coefficients, with shape: [2N-1, L, 2L-1].
        kernel (jnp.ndarray): Wigner transform kernel.
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Pixel-space coefficients with shape [2N-1, L, 2L-1].

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    fnlm_shift = jnp.fft.ifftshift(fnlm, axes=-1)
    fnam = jnp.einsum("...lmi, ...lm->...im", kernel, fnlm_shift, optimize=True)
    fnab = jnp.fft.ifft(fnam, axis=-1, norm="forward")
    return jnp.fft.ifft(jnp.fft.ifftshift(fnab, axes=-3), norm="forward", axis=-3)
