import numpy as np


def generate_flm(
    rng: np.random.Generator,
    L: int,
    L_lower: int = 0,
    spin: int = 0,
    reality: bool = False,
) -> np.ndarray:
    r"""Generate a 2D set of random harmonic coefficients.

    Note:
        Real signals are explicitly produced from conjugate symmetry.

    Args:
        rng (Generator): Random number generator.
        L (int): Harmonic band-limit.
        L_lower (int, optional): Harmonic lower bound. Defaults to 0.
        spin (int, optional): Harmonic spin. Defaults to 0.
        reality (bool, optional): Reality of signal. Defaults to False.

    Returns:
        np.ndarray: Random set of spherical harmonic coefficients.
    """
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)

    for el in range(max(L_lower, abs(spin)), L):

        if reality:
            flm[el, 0 + L - 1] = rng.uniform()
        else:
            flm[el, 0 + L - 1] = rng.uniform() + 1j * rng.uniform()

        for m in range(1, el + 1):
            flm[el, m + L - 1] = rng.uniform() + 1j * rng.uniform()
            if reality:
                flm[el, -m + L - 1] = (-1) ** m * np.conj(flm[el, m + L - 1])
            else:
                flm[el, -m + L - 1] = rng.uniform() + 1j * rng.uniform()

    return flm


def generate_flmn(
    rng: np.random.Generator,
    L: int,
    N: int = 1,
    L_lower: int = 0,
    reality: bool = False,
) -> np.ndarray:
    r"""Generate a 3D set of random Wigner coefficients.

    Note:
        Real signals are explicitly produced from conjugate symmetry.

    Args:
        rng (Generator): Random number generator.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        L_lower (int, optional): Harmonic lower bound. Defaults to 0.
        reality (bool, optional): Reality of signal. Defaults to False.

    Returns:
        np.ndarray: Random set of Wigner coefficients.
    """
    flmn = np.zeros((2 * N - 1, L, 2 * L - 1), dtype=np.complex128)

    for n in range(-N + 1, N):

        for el in range(max(L_lower, abs(n)), L):

            if reality:
                flmn[N - 1 + n, el, 0 + L - 1] = rng.uniform()
            else:
                flmn[N - 1 + n, el, 0 + L - 1] = (
                    rng.uniform() + 1j * rng.uniform()
                )

            for m in range(1, el + 1):
                flmn[N - 1 + n, el, m + L - 1] = (
                    rng.uniform() + 1j * rng.uniform()
                )
                if reality:
                    flmn[N - 1 + n, el, -m + L - 1] = (-1) ** m * np.conj(
                        flmn[N - 1 + n, el, m + L - 1]
                    )
                else:
                    flmn[N - 1 + n, el, -m + L - 1] = (
                        rng.uniform() + 1j * rng.uniform()
                    )

    return flmn


def generate_flmp(rng: np.random.Generator, L: int, P: int) -> np.ndarray:
    r"""Generate a 3D set of random Spherical-Laguerre coefficients.

    Args:
        rng (Generator): Random number generator.
        L (int): Harmonic band-limit.
        P (int): Radial band-limit.

    Returns:
        np.ndarray: Random set of Spherical-Laguerre coefficients.
    """
    s = (P, L, 2 * L - 1)
    return rng.uniform(size=s) + 1j * rng.uniform(size=s)


def generate_flmnp(
    rng: np.random.Generator, L: int, N: int, P: int
) -> np.ndarray:
    r"""Generate a 4D set of random Wigner-Laguerre coefficients.

    Args:
        rng (Generator): Random number generator.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit. Must be < L.
        P (int): Radial band-limit.

    Returns:
        np.ndarray: Random set of Spherical-Laguerre coefficients.
    """
    s = (P, 2 * N - 1, L, 2 * L - 1)
    return rng.uniform(size=s) + 1j * rng.uniform(size=s)


def flm_2d_to_1d(flm_2d: np.ndarray, L: int) -> np.ndarray:
    r"""Convert from 2D indexed harmonic coefficients to 1D indexed coefficients.
    
    Note:
        Storage conventions for harmonic coefficients :math:`flm_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::

            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{1D data format}:  [flm_{0,0}, flm_{1,-1}, flm_{1,0}, flm_{1,1}, \dots]

    Args:
        flm_2d (np.ndarray): 2D indexed harmonic coefficients.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: 1D indexed harmonic coefficients.
    """
    flm_1d = np.zeros(ncoeff(L), dtype=np.complex128)

    if len(flm_2d.shape) != 2:
        if len(flm_2d.shape) == 1:
            raise ValueError(f"Flm is already 1D indexed")
        else:
            raise ValueError(
                f"Cannot convert flm of dimension {flm_2d.shape} to 1D indexing"
            )

    for el in range(L):
        for m in range(-el, el + 1):
            flm_1d[elm2ind(el, m)] = flm_2d[el, L - 1 + m]

    return flm_1d


def flm_1d_to_2d(flm_1d: np.ndarray, L: int) -> np.ndarray:
    r"""Convert from 1D indexed harmnonic coefficients to 2D indexed coefficients.    
    
    Note:
        Storage conventions for harmonic coefficients :math:`flm_{(\ell,m)}`, for 
        e.g. :math:`L = 3`, are as follows.

        .. math::

            \text{ 2D data format}:
                \begin{bmatrix}
                    0 & 0 & flm_{(0,0)} & 0 & 0 \\
                    0 & flm_{(1,-1)} & flm_{(1,0)} & flm_{(1,1)} & 0 \\
                    flm_{(2,-2)} & flm_{(2,-1)} & flm_{(2,0)} & flm_{(2,1)} & flm_{(2,2)}
                \end{bmatrix}
        
        .. math::

            \text{1D data format}:  [flm_{0,0}, flm_{1,-1}, flm_{1,0}, flm_{1,1}, \dots]

    Args:
        flm_1d (np.ndarray): 1D indexed harmonic coefficients.

        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: 2D indexed harmonic coefficients.
    """

    flm_2d = np.zeros((L, 2 * L - 1), dtype=np.complex128)

    if len(flm_1d.shape) != 1:
        if len(flm_1d.shape) == 2:
            raise ValueError(f"Flm is already 2D indexed")
        else:
            raise ValueError(
                f"Cannot convert flm of dimension {flm_2d.shape} to 2D indexing"
            )

    for el in range(L):
        for m in range(-el, el + 1):
            flm_2d[el, L - 1 + m] = flm_1d[elm2ind(el, m)]

    return flm_2d


def elm2ind(el: int, m: int) -> int:
    """Convert from spherical harmonic 2D indexing of :math:`(\ell,m)` to 1D index.

    1D index is defined by `el**2 + el + m`.

    Warning:
        Note that 1D storage of spherical harmonic coefficients is *not* the default.

    Args:
        el (int): Harmonic degree :math:`\ell`.

        m (int): Harmonic order :math:`m`.

    Returns:
        int: Corresponding 1D index value.
    """

    return el**2 + el + m


def ind2elm(ind: int) -> tuple:
    """Convert from 1D spherical harmonic index to 2D index of :math:`(\ell,m)`.

    Warning:
        Note that 1D storage of spherical harmonic coefficients is *not* the default.

    Args:
        ind (int): 1D spherical harmonic index.

    Returns:
        tuple: `(el,m)` defining spherical harmonic degree and order.
    """

    el = np.floor(np.sqrt(ind))

    m = ind - el**2 - el

    return el, m


def ncoeff(L: int) -> int:
    """Number of spherical harmonic coefficients for given band-limit L.

    Args:
        L (int, optional): Harmonic band-limit.

    Returns:
        int: Number of spherical harmonic coefficients.
    """

    return elm2ind(L - 1, L - 1) + 1


def elmn2ind(el: int, m: int, n: int, L: int, N: int) -> int:
    """Convert from Wigner space 3D indexing of :math:`(\ell,m, n)` to 1D index.
    Args:
        el (int): Harmonic degree :math:`\ell`.
        m (int): Harmonic order :math:`m`.
        n (int): Directional order :math:`n`.
        L (int): Harmonic band-limit.
        N (int, optional): Directional band-limit. Defaults to 1.
    Returns:
        int: Corresponding 1D index in Wigner space.
    """
    n_offset = (N - 1 + n) * L * L
    el_offset = el * el
    return n_offset + el_offset + el + m


def flmn_3d_to_1d(flmn_3d: np.ndarray, L: int, N: int) -> np.ndarray:
    r"""Convert from 3D indexed Wigner coefficients to 1D indexed coefficients.
    Args:
        flm_3d (np.ndarray): 3D indexed Wigner coefficients, index order
            :math:`[n, \ell, m]`.
        L (int): Harmonic band-limit.
        N (int, optional): Directional band-limit.
    Raises:
        ValueError: `flmn` is already 1D indexed.
        ValueError: `flmn` is not 3D.
    Returns:
        np.ndarray: 1D indexed Wigner coefficients, C flatten index priority :math:`n, \ell, m`.
    """
    flmn_1d = np.zeros((2 * N - 1) * L * L, dtype=np.complex128)

    if len(flmn_3d.shape) == 1:
        raise ValueError(f"flmn is already 1D indexed")
    elif len(flmn_3d.shape) != 3:
        raise ValueError(
            f"Cannot convert flmn of dimension {flmn_3d.shape} to 1D indexing"
        )

    for n in range(-N + 1, N):
        for el in range(L):
            for m in range(-el, el + 1):
                flmn_1d[elmn2ind(el, m, n, L, N)] = flmn_3d[
                    N - 1 + n, el, L - 1 + m
                ]

    return flmn_1d


def flmn_1d_to_3d(flmn_1d: np.ndarray, L: int, N: int) -> np.ndarray:
    r"""Convert from 1D indexed Wigner coefficients to 3D indexed coefficients.
    Args:
        flm_1d (np.ndarray): 1D indexed Wigner coefficients, C flatten index priority
            :math:`n, \ell, m`.
        L (int): Harmonic band-limit.
        N (int): Directional band-limit.
    Raises:
        ValueError: `flmn` is already 3D indexed.
        ValueError: `flmn` is not 1D.
    Returns:
        np.ndarray: 3D indexed Wigner coefficients, index order :math:`[n, \ell, m]`.
    """
    flmn_3d = np.zeros((2 * N - 1, L, 2 * L - 1), dtype=np.complex128)

    if len(flmn_1d.shape) == 3:
        raise ValueError(f"Flmn is already 3D indexed")
    elif len(flmn_1d.shape) != 1:
        raise ValueError(
            f"Cannot convert flmn of dimension {flmn_1d.shape} to 3D indexing"
        )

    for n in range(-N + 1, N):
        for el in range(L):
            for m in range(-el, el + 1):
                flmn_3d[N - 1 + n, el, L - 1 + m] = flmn_1d[
                    elmn2ind(el, m, n, L, N)
                ]

    return flmn_3d
