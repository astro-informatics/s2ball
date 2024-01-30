import numpy as np
from typing import Tuple


def polynomials(P: int, tau: float, forward: bool = True) -> np.ndarray:
    """Computes Laguerre polynomials, sampled at nodes coincident with zeros.

    Args:
        P (int): Radial band-limit.
        tau (float): Laguerre polynomial scale factor.
        forward (bool, optional): Whether to load the forward or inverse matrices.
            Defaults to True.

    Returns:
        np.ndarray: Array containing Laguerre polynomials sampled at zeros.
    """

    nodes, weights = nodes_and_weights(P, tau)
    if not forward:
        weights = None

    temp = np.zeros((len(nodes), P), dtype=np.float64)

    for n, node in enumerate(nodes):

        weight = 1 if weights is None else weights[n]
        r = node / tau
        factor = weight * np.exp(-r / 4.0)
        lagu0 = 0.0
        lagu1 = 1.0 * np.exp(-r / 4.0)

        temp[n, 0] = factor * lagu1

        for p in range(1, P):
            lagu2 = ((2 * p - 1 - r) * lagu1 - (p - 1) * lagu0) / p

            temp[n, p] = factor * lagu2

            lagu0 = lagu1
            lagu1 = lagu2

    return temp


def nodes_and_weights(P: int, tau: float) -> Tuple[np.ndarray]:
    r"""Evaluates nodes and corresponding weights for zeros of the Laguerre polynomials.

    Args:
        P (int): Radial band-limit.
        tau (float): Laguerre polynomial scale factor.

    Returns:
        Tuple[np.ndarray]: Nodes scaled by :math:`\tau` and corresponding weights.
    """
    nodes, weights = quadrature(P)
    return tau * nodes, weights


def quadrature(P: int) -> np.ndarray:
    """Compute Gauss-Laguerre quadrature over the positive half-line.

    Args:
        P (int): Radial band-limit.

    Returns:
        np.ndarray: Gauss-Laguerre quadrature weights.
    """
    nodes = np.zeros(P, dtype=np.float64)
    weights = np.zeros(P, dtype=np.float64)

    nitermax = 2000
    maxit = 251
    z = 0.0

    h = 1.0 / float(P)
    normfac = 1.0
    infbound = h

    for p in range(P):
        if p > 100:
            h = 0.1
        supbound = infbound
        normfac = _laguerre_rescaled(z, p, normfac)
        temp = _laguerre_rescaled(infbound, P, normfac)
        vinf = vsup = temp

        niter = 0
        while vinf * vsup >= 0 and niter < nitermax:
            supbound += h
            vsup = _laguerre_rescaled(supbound, P, normfac)
            niter += 1

        niter = 0
        while vinf * vsup < 0 and niter < nitermax:
            infbound += h
            vinf = _laguerre_rescaled(infbound, P, normfac)
            niter += 1

        infbound -= h
        vinf = _laguerre_rescaled(infbound, P, normfac)
        z = infbound - vinf * (supbound - infbound) / (vsup - vinf)

        infbound = supbound
        for i in range(1, maxit):
            p1 = _laguerre_rescaled(z, P, normfac)
            p2 = _laguerre_rescaled(z, P - 1, normfac)

            pp = P * (p1 - p2) / z
            z1 = z
            z = z1 - p1 / pp

            if abs(z - z1) < 1e-16:
                break

        nodes[p] = z

        denom = normfac * _laguerre_rescaled(z, P + 1, normfac)
        term = np.exp(z / 2.0) / denom
        weights[p] = z * (term / (P + 1)) ** 2

    return nodes, weights


def _laguerre_rescaled(z: float, P: int, normfac: float) -> float:
    """Helper function for Gauss-Laguerre quadrature."""
    p1 = 1.0 / normfac
    p2 = 0.0
    for p in range(1, P + 1):
        p3 = p2
        p2 = p1
        p1 = ((2.0 * p - 1.0 - z) * p2 - (p - 1.0) * p3) / p
    return p1
