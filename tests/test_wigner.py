import os
import numpy as np
import so3
import pytest

from baller.transform import wigner
from baller.construct.wigner_constructor import *
from baller import utils

L_to_test = [8, 16, 32]
N_to_test = [2, 4, 6]
methods_to_test = ["numpy", "jax"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
def test_forward_wigner_constructor(L: int, N: int):
    """Test creation and saving down of forward Wigner kernels"""
    save_dir = ".matrices"
    filename = save_dir + "/wigner_matrix_{}_N_{}.npy".format(L, N)

    wigner_forward = construct_wigner_matrix(L, N, save_dir)
    assert wigner_forward.shape == (2 * N - 1, L, 2 * L - 1, L)
    assert os.path.isfile(filename)
    wigner_forward_check = load_wigner_matrix(L, N, save_dir, forward=True)
    np.testing.assert_allclose(wigner_forward, wigner_forward_check, rtol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
def test_inverse_wigner_constructor(L: int, N: int):
    """Test creation and saving down of inverse Wigner kernels"""
    save_dir = ".matrices"
    filename = save_dir + "/wigner_inverse_matrix_{}_N_{}.npy".format(L, N)

    wigner_inverse = construct_wigner_matrix_inverse(L, N, save_dir)
    assert wigner_inverse.shape == (2 * N - 1, L, 2 * L - 1, L)
    assert os.path.isfile(filename)
    wigner_inverse_check = load_wigner_matrix(L, N, save_dir, forward=False)
    np.testing.assert_allclose(wigner_inverse, wigner_inverse_check, rtol=1e-14)


def test_legendre_matrix_constructor_compute_time_warning():
    """Test compile time warning of legendre kernels"""
    compute_wigner_warning(128, 1)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_wigner_forward(flmn_generator, L: int, N: int, method: str):
    """Test wrapper implementation of forward wigner transform"""
    save_dir = ".matrices"
    params = so3.create_parameter_dict(
        L=L, N=N, sampling_scheme_str="SO3_SAMPLING_MW"
    )

    wig_for = construct_wigner_matrix(L, N, save_dir)

    flmn_3d = flmn_generator(L=L, N=N)
    flmn_1d = utils.flmn_3d_to_1d(flmn_3d, L, N)

    f_1d = so3.inverse(flmn_1d, params)
    f_3d = f_1d.reshape(2 * N - 1, L, 2 * L - 1)

    flmn_check = utils.flmn_1d_to_3d(so3.forward(f_1d, params), L, N)
    flmn = wigner.forward(f_3d, L, N, wig_for, method)

    assert np.allclose(flmn, flmn_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_wigner_inverse(flmn_generator, L: int, N: int, method: str):
    """Test wrapper implementation of inverse wigner transform"""
    save_dir = ".matrices"
    params = so3.create_parameter_dict(
        L=L, N=N, sampling_scheme_str="SO3_SAMPLING_MW"
    )
    wig_inv = construct_wigner_matrix_inverse(L, N, save_dir)

    flmn_3d = flmn_generator(L=L, N=N)
    flmn_1d = utils.flmn_3d_to_1d(flmn_3d, L, N)

    f_check = so3.inverse(flmn_1d, params).reshape(2 * N - 1, L, 2 * L - 1)
    f = wigner.inverse(flmn_3d, L, N, wig_inv, method)

    assert np.allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_wigner_forward_adjoint(flmn_generator, L: int, N: int, method: str):
    """Test wrapper implementation of forward adjoint wigner transform"""

    flmn = flmn_generator(L=L, N=N)
    f = wigner.inverse(flmn, L, N, None, method)

    flmn_forward = wigner.forward(f, L, N, None, method)
    f_adjoint = wigner.forward(flmn, L, N, None, method, adjoint=True)

    a = np.abs(np.vdot(f, f_adjoint))
    b = np.abs(np.vdot(flmn, flmn_forward))
    assert a == pytest.approx(b)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("method", methods_to_test)
def test_wigner_inverse_adjoint(flmn_generator, L: int, N: int, method: str):
    """Test wrapper implementation of inverse adjoint wigner transform"""

    flmn = flmn_generator(L=L, N=N)
    f = wigner.inverse(flmn, L, N, method=method)

    flmn_inverse_adjoint = wigner.inverse(f, L, N, method=method, adjoint=True)
    f_inverse = wigner.inverse(flmn, L, N, method=method)

    a = np.abs(np.vdot(f, f_inverse))
    b = np.abs(np.vdot(flmn, flmn_inverse_adjoint))
    assert a == pytest.approx(b)
