import numpy as np
import warnings
import os
from baller.construct.legendre_constructor import (
    load_legendre_matrix,
)


def construct_wigner_matrix(
    L: int, N: int, save_dir: str = ".matrices"
) -> np.ndarray:
    """Construct Wigner matrix which will be called during transform.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".

    Returns:
        np.ndarray: Wigner matrix elements for forward Wigner transform.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    Wigner = np.zeros((2 * N - 1, L, 2 * L - 1, L), dtype=np.float64)

    for n in range(-N + 1, N):
        Wigner[N - 1 + n, ...] = load_legendre_matrix(
            L, save_dir, forward=True, spin=-n
        )
        for el in range(L):
            Wigner[N - 1 + n, el, ...] *= (-1) ** n * np.sqrt(
                4.0 * np.pi / (2.0 * el + 1)
            )

    if save_dir:
        if not os.path.isdir("{}/".format(save_dir)):
            os.mkdir(save_dir)
        save_dir = save_dir + "/"
        filename = "{}wigner_matrix_{}_N_{}".format(save_dir, L, N)
    else:
        filename = "wigner_matrix_{}_N_{}".format(L, N)

    np.save(filename, Wigner)
    return Wigner


def construct_wigner_matrix_inverse(
    L: int, N: int, save_dir: str = ".matrices"
) -> np.ndarray:
    """Construct associated Legendre inverse matrix for precompute method.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".

    Returns:
        np.ndarray: Associated Legendre matrix for inverse harmonic transform.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """

    compute_wigner_warning(L, N)
    Wigner_inverse = np.zeros((2 * N - 1, L, 2 * L - 1, L), dtype=np.float64)

    for n in range(-N + 1, N):
        Wigner_inverse[N - 1 + n, ...] = load_legendre_matrix(
            L, save_dir, forward=False, spin=-n
        )
        for el in range(L):
            Wigner_inverse[N - 1 + n, el, ...] *= (-1) ** n * np.sqrt(
                (2.0 * el + 1) / (16 * np.pi**3)
            )

    if save_dir:
        if not os.path.isdir("{}/".format(save_dir)):
            os.mkdir(save_dir)
        save_dir = save_dir + "/"
        filename = "{}wigner_inverse_matrix_{}_N_{}".format(save_dir, L, N)
    else:
        filename = "wigner_inverse_matrix_{}_N_{}".format(L, N)

    np.save(filename, Wigner_inverse)
    return Wigner_inverse


def compute_wigner_warning(L: int, N: int):
    r"""Basic compute warning for large Legendre precomputes.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.

    Raises:
        Warning: If the estimated time for precompute is large (:math:`L>=128`).
    """

    base_value = 10
    if L >= 128:
        warnings.warn(
            f"Wigner matrix precomputation currently scales as NL^5. \
            Estimated compilation time is ~{N * (base_value * (L / 128) ** 2) / 60.0} hours\
            Get a coffee this may take a moment."
        )


def load_wigner_matrix(
    L: int, N: int, save_dir: str = ".matrices", forward: bool = True
) -> np.ndarray:
    """Construct associated Legendre inverse matrix for precompute method.

    Args:
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        forward (bool, optional): Whether to load the forward or inverse matrices.
            Defaults to True.

    Returns:
        np.ndarray: Associated Legendre matrix for corresponding harmonic transform.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1508.03101.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """

    dir_string = ""
    if not forward:
        dir_string += "_inverse"

    filepath = "{}/wigner{}_matrix_{}_N_{}.npy".format(
        save_dir, dir_string, L, N
    )

    if not os.path.isfile(filepath):
        if forward:
            construct_wigner_matrix(
                L=L,
                N=N,
                save_dir=save_dir,
            )
        else:
            construct_wigner_matrix_inverse(
                L=L,
                N=N,
                save_dir=save_dir,
            )
    return np.load(filepath)
