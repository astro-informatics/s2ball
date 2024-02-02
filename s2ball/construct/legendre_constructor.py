import numpy as np
import pyssht as ssht
import os
import warnings


def construct_legendre_matrix(
    L: int, save_dir: str = ".matrices", spin: int = 0
) -> np.ndarray:
    """Construct associated Legendre matrix which will be called during transform.

    Args:
        L (int): Harmonic band-limit.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Associated Legendre matrix for forward harmonic transform.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """
    Legendre = np.zeros((L * L, L), dtype=np.float64)

    for i in range(L):

        in_matrix = np.zeros((L, 2 * L - 1), dtype=np.complex128)
        in_matrix[i, 0] = 1.0

        Legendre[:, i] = np.real(
            ssht.forward(f=in_matrix, L=L, Method="MW", Spin=spin).flatten("C")
        )

    Legendre_reshaped = np.zeros((L, 2 * L - 1, L), dtype=np.float64)

    for l in range(L):
        for m in range(-l, l + 1):
            if m < 0:
                ind_tf = Legendre_reshaped.shape[1] + m
            if m >= 0:
                ind_tf = m
            ind_ssht = ssht.elm2ind(l, m)
            Legendre_reshaped[l, ind_tf, :] = Legendre[ind_ssht, :]
    Legendre = Legendre_reshaped

    if save_dir:
        if not os.path.isdir("{}/".format(save_dir)):
            os.mkdir(save_dir)
        save_dir = save_dir + "/"
        filename = "{}legendre_matrix_{}_spin_{}".format(save_dir, L, spin)
    else:
        filename = "legendre_matrix_{}_spin_{}".format(L, spin)

    np.save(filename, Legendre)
    return Legendre_reshaped


def construct_legendre_matrix_inverse(
    L: int, save_dir: str = ".matrices", spin: int = 0
) -> np.ndarray:
    """Construct associated Legendre inverse matrix for precompute method.

    Args:
        L (int): Harmonic band-limit.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        spin (int, optional): Harmonic spin. Defaults to 0.

    Returns:
        np.ndarray: Associated Legendre matrix for inverse harmonic transform.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """

    compute_legendre_warning(L)

    Legendre_inverse = np.zeros((L * L, L), dtype=np.float64)
    alm = np.zeros(L * L, dtype=np.complex128)

    for l in range(L):
        for m in range(-l, l + 1):
            ind = ssht.elm2ind(l, m)
            alm[:] = 0.0
            alm[ind] = 1.0
            Legendre_inverse[ind, :] = np.real(
                ssht.inverse(flm=alm, L=L, Method="MW", Spin=spin)[:, 0]
            )

    Legendre_reshaped = np.zeros((L, 2 * L - 1, L), dtype=np.float64)

    for l in range(L):
        for m in range(-l, l + 1):
            if m < 0:
                ind_tf = Legendre_reshaped.shape[1] + m
            if m >= 0:
                ind_tf = m
            ind_ssht = ssht.elm2ind(l, m)
            Legendre_reshaped[l, ind_tf, :] = Legendre_inverse[ind_ssht, :]
    Legendre_inverse = Legendre_reshaped

    if save_dir:
        if not os.path.isdir("{}/".format(save_dir)):
            os.mkdir(save_dir)
        save_dir = save_dir + "/"
        filename = "{}legendre_inverse_matrix_{}_spin_{}".format(save_dir, L, spin)
    else:
        filename = "legendre_inverse_matrix_{}_spin_{}".format(L, spin)

    np.save(filename, Legendre_inverse)

    return Legendre_inverse


def compute_legendre_warning(L):
    r"""Basic compute warning for large Legendre precomputes.

    Args:
        L (int): Harmonic band-limit.

    Raises:
        Warning: If the estimated time for precompute is large (:math:`L>=256`).
    """

    base_value = 10
    if L >= 256:
        warnings.warn(
            f"Inverse associated Legendre matrix precomputation currently scales as L^5. \
            Estimated compilation time is ~{(base_value * (L / 128) ** 2) / 60.0} hours\
            Get a coffee this may take a moment."
        )


def load_legendre_matrix(
    L: int, save_dir: str = ".matrices", forward: bool = True, spin: int = 0
) -> np.ndarray:
    """Load/construct associated Legendre inverse matrix for precompute method.

    Args:
        L (int): Harmonic band-limit.
        save_dir (str, optional): Directory in which to save precomputed matrices.
            Defaults to ".matrices".
        forward (bool, optional): Whether to load the forward or inverse matrices.
            Defaults to True.
        spin (int, optional): Spin of the transform to consider. Defaults to 0.

    Returns:
        np.ndarray: Associated Legendre matrix for corresponding harmonic transform.

    Note:
        Currently only `McEwen-Wauix <https://arxiv.org/pdf/1110.6298.pdf>`_
        sampling on the sphere is supported, though this approach can be
        extended to alternate sampling schemes, e.g. HEALPix.
    """

    dir_string = ""
    if not forward:
        dir_string += "_inverse"

    filepath = "{}/legendre{}_matrix_{}_spin_{}.npy".format(
        save_dir, dir_string, L, spin
    )

    if not os.path.isfile(filepath):
        if forward:
            construct_legendre_matrix(
                L=L,
                save_dir=save_dir,
                spin=spin,
            )
        else:
            construct_legendre_matrix_inverse(
                L=L,
                save_dir=save_dir,
                spin=spin,
            )
    return np.load(filepath)
