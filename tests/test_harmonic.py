from jax.config import config

config.update("jax_enable_x64", True)

import os
import numpy as np
import pyssht as ssht
import pytest
from s2ball.transform import harmonic
from s2ball.construct.legendre_constructor import *
from s2ball.construct import matrix
from s2ball import utils

L_to_test = [16, 32]
spin_to_test = [-2, 0, 2]
methods_to_test = ["numpy", "jax"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
def test_forward_legendre_matrix_constructor(L: int, spin: int):
    """Test creation and saving down of forward Legendre kernels"""
    save_dir = ".matrices"
    filename = save_dir + "/legendre_matrix_{}_spin_{}.npy".format(L, spin)

    legendre_forward = construct_legendre_matrix(L, save_dir, spin)
    assert legendre_forward.shape == (L, 2 * L - 1, L)
    assert os.path.isfile(filename)
    legendre_forward = construct_legendre_matrix(L, save_dir, spin)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
def test_inverse_legendre_matrix_constructor(L: int, spin: int):
    """Test creation and saving down of inverse Legendre kernels"""
    save_dir = ".matrices"
    filename = save_dir + "/legendre_inverse_matrix_{}_spin_{}.npy".format(L, spin)

    legendre_inverse = construct_legendre_matrix_inverse(L, save_dir, spin)
    assert legendre_inverse.shape == (L, 2 * L - 1, L)
    assert os.path.isfile(filename)
    legendre_inverse = construct_legendre_matrix_inverse(L, save_dir, spin)


def test_legendre_matrix_constructor_compute_time_warning():
    """Test compile time warning of legendre kernels"""
    compute_legendre_warning(256)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_transform_precompute(flm_generator, L: int, spin: int, method: str):
    """Test wrapper implementation of forward/inverse precompute sht"""
    save_dir = ".matrices"

    legendre = matrix.generate_matrices(
        transform="spherical_harmonic", L=L, spin=spin, save_dir=save_dir
    )
    flm = flm_generator(L=L, spin=spin)

    f = ssht.inverse(utils.flm_2d_to_1d(flm, L), L, spin)

    flm_precomp = harmonic.forward(f, L, legendre, method, spin)
    assert np.allclose(flm_precomp, flm)

    f_precomp = harmonic.inverse(flm_precomp, L, legendre, method, spin)
    assert np.allclose(f_precomp, f)


def test_transform_device_exceptions():
    """Test device exception of forward/inverse precompute sht"""
    f = ["not a vector"]
    legendre_kernel = ["not a kernel"]
    device = ["not a device"]
    with pytest.raises(ValueError):
        harmonic.forward(f, 8, legendre_kernel, device, 0)
    with pytest.raises(ValueError):
        harmonic.inverse(f, 8, legendre_kernel, device, 0)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_transform_precompute_load_legendre(
    flm_generator, L: int, spin: int, method: str
):
    """Test wrapper implementation of forward/inverse precompute sht"""
    save_dir = ".matrices"

    flm = flm_generator(L=L, spin=spin)

    f = ssht.inverse(utils.flm_2d_to_1d(flm, L), L, spin)

    flm_precomp = harmonic.forward(f, L=L, method=method, spin=spin, save_dir=save_dir)
    assert np.allclose(flm_precomp, flm)

    f_precomp = harmonic.inverse(
        flm_precomp, L=L, method=method, spin=spin, save_dir=save_dir
    )
    assert np.allclose(f_precomp, f)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
def test_transform_precompute_numpy(flm_generator, L: int, spin: int):
    """Test cpu implementation of forward/inverse precompute sht"""
    save_dir = ".matrices"

    legendre = matrix.generate_matrices(
        transform="spherical_harmonic", L=L, spin=spin, save_dir=save_dir
    )
    flm = flm_generator(L=L, spin=spin)

    f = ssht.inverse(utils.flm_2d_to_1d(flm, L), L, spin)

    flm_test = harmonic.forward_transform(f, legendre)
    assert np.allclose(flm_test, flm)
    f_test = harmonic.inverse_transform(flm_test, legendre)
    assert np.allclose(f_test, f)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
def test_transform_precompute_jax(flm_generator, L: int, spin: int):
    """Test gpu implementation of forward/inverse precompute sht"""
    save_dir = ".matrices"

    legendre = matrix.generate_matrices(
        transform="spherical_harmonic", L=L, spin=spin, save_dir=save_dir
    )
    flm = flm_generator(L=L, spin=spin, reality=False)

    f = ssht.inverse(utils.flm_2d_to_1d(flm, L), L, spin)

    flm_test = harmonic.forward_transform_jax(f, legendre)
    assert np.allclose(flm_test, flm)
    f_test = harmonic.inverse_transform_jax(flm_test, legendre)
    assert np.allclose(f_test, f)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_forward_adjoint_transform(flm_generator, L: int, spin: int, method: str):
    """Test wrapper implementation of forward adjoint sht"""
    flm = flm_generator(L=L, spin=spin)
    f = ssht.inverse(utils.flm_2d_to_1d(flm, L), L, spin)

    flm_forward = harmonic.forward(f, L, None, method, spin)
    f_adjoint = harmonic.forward(flm, L, None, method, spin, adjoint=True)

    a = np.abs(np.vdot(f, f_adjoint))
    b = np.abs(np.vdot(flm, flm_forward))
    assert a == pytest.approx(b)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("spin", spin_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_inverse_adjoint_transform(flm_generator, L: int, spin: int, method: str):
    """Test wrapper implementation of inverse adjoint sht"""
    flm = flm_generator(L=L, spin=spin)
    f = ssht.inverse(utils.flm_2d_to_1d(flm, L), L, spin)

    flm_inverse_adjoint = harmonic.inverse(f, L, None, method, spin, adjoint=True)
    f_inverse = harmonic.inverse(flm, L, None, method, spin)

    a = np.abs(np.vdot(f, f_inverse))
    b = np.abs(np.vdot(flm, flm_inverse_adjoint))
    assert a == pytest.approx(b)
