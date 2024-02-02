import numpy as np
from scipy.special import loggamma
import math

# Note that these functions are ported from s2wav (https://github.com/astro-informatics/s2wav).


def tiling_integrand(t: float, lam: float = 2.0) -> float:
    r"""Tiling integrand for scale-discretised wavelets `[1] <https://arxiv.org/pdf/1211.1680.pdf>`_.
    Intermediate step used to compute the wavelet and scaling function generating
    functions. One of the basic mathematical functions needed to carry out the tiling of
    the harmonic space.
    Args:
        t (float): Real argument over which we integrate.
        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales.Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
    Returns:
        float: Value of tiling integrand for given :math:`t` and scaling factor.
    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """
    s_arg = (t - (1 / lam)) * (2.0 * lam / (lam - 1)) - 1

    integrand = np.exp(-2.0 / (1.0 - s_arg**2.0)) / t

    return integrand


def part_scaling_fn(a: float, b: float, n: int, lam: float = 2.0) -> float:
    r"""Computes integral used to calculate smoothly decreasing function :math:`k_{\lambda}`.
    Intermediate step used to compute the wavelet and scaling function generating
    functions. Uses the trapezium method to integrate :func:`~tiling_integrand` in the
    limits from :math:`a \rightarrow b` with scaling parameter :math:`\lambda`. One of
    the basic mathematical functions needed to carry out the tiling of the harmonic
    space.
    Args:
        a (float): Lower limit of the numerical integration.
        b (float): Upper limit of the numerical integration.
        n (int): Number of steps to be performed during integration.
        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales.Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
    Returns:
        float: Integral of the tiling integrand from :math:`a \rightarrow b`.
    """
    sum = 0.0
    h = (b - a) / n

    if a == b:
        return 0

    for i in range(n):
        if a + i * h not in [1 / lam, 1.0] and a + (i + 1) * h not in [
            1 / lam,
            1.0,
        ]:
            f1 = tiling_integrand(a + i * h, lam)
            f2 = tiling_integrand(a + (i + 1) * h, lam)

            sum += ((f1 + f2) * h) / 2

    return sum


def k_lam(L: int, lam: float = 2.0, quad_iters: int = 300) -> float:
    r"""Compute function :math:`k_{\lambda}` used as a wavelet generating function.
    Specifically, this function is derived in [1] and is given by

    .. math::

        k_{\lambda} \equiv \frac{ \int_t^1 \frac{\text{d}t^{\prime}}{t^{\prime}}
        s_{\lambda}^2(t^{\prime})}{ \int_{\frac{1}{\lambda}}^1
        \frac{\text{d}t^{\prime}}{t^{\prime}} s_{\lambda}^2(t^{\prime})},

    where the integrand is defined to be

    .. math::

        s_{\lambda} \equiv s \Big ( \frac{2\lambda}{\lambda - 1}(t-\frac{1}{\lambda})
        - 1 \Big ),

    for infinitely differentiable Cauchy-Schwartz function :math:`s(t) \in C^{\infty}`.

    Args:
        L (int): Harmonic band-limit.
        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.
        quad_iters (int, optional): Total number of iterations for quadrature
            integration. Defaults to 300.
    Returns:
        (np.ndarray): Value of :math:`k_{\lambda}` computed for values between
            :math:`\frac{1}{\lambda}` and 1, parametrised by :math:`\ell` as required to
            compute the axisymmetric filters in :func:`~tiling_axisym`.
    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """

    J = math.ceil(np.log(L) / np.log(lam))

    normalisation = part_scaling_fn(1 / lam, 1.0, quad_iters, lam)
    k = np.zeros((J + 2, L), dtype=np.float64)

    for j in range(J + 2):
        for l in range(L):
            if l < lam ** (j - 1):
                k[j, l] = 1
            elif l > lam**j:
                k[j, l] = 0
            else:
                k[j, l] = (
                    part_scaling_fn(l / lam**j, 1.0, quad_iters, lam) / normalisation
                )

    return k


def tiling_direction(L: int, N: int) -> np.ndarray:
    r"""Generates the harmonic coefficients for the directionality component of the
        tiling functions.

    Args:
        L (int): Harmonic band-limit.
        N (int): Upper orientational band-limit.
    Returns:
        np.ndarray: Harmonic coefficients of directionality components
            :math:`_{s}\eta_{\ell m}`.
    Notes:
        [1] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    if N % 2:
        nu = 1
    else:
        nu = 1j

    s_elm = np.zeros((L, 2 * L - 1), dtype=np.complex128)

    for el in range(1, L):
        if (N + el) % 2:
            gamma = min(N - 1, el)
        else:
            gamma = min(N - 1, el - 1)

        for m in range(-el, el + 1):
            if abs(m) < N and (N + m) % 2:
                s_elm[el, L - 1 + m] = nu * np.sqrt(
                    (binomial_coefficient(gamma, ((gamma - m) / 2))) / (2**gamma)
                )
            else:
                s_elm[el, L - 1 + m] = 0.0

    return s_elm


def binomial_coefficient(n: int, k: int) -> int:
    r"""Computes the binomial coefficient :math:`\binom{n}{k}`.
    Args:
        n (int): Number of elements to choose from.
        k (int): Number of elements to pick.
    Returns:
        (int): Number of possible subsets.
    """
    return np.floor(
        0.5 + np.exp(loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1))
    )
