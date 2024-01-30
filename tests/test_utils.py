import numpy as np
import pytest
import s2ball


def test_flm_reindexing_functions(flm_generator):

    L = 16
    flm_2d = flm_generator(L=L, spin=0, reality=False)

    flm_1d = s2ball.utils.flm_2d_to_1d(flm_2d, L)

    assert len(flm_1d.shape) == 1

    flm_2d_check = s2ball.utils.flm_1d_to_2d(flm_1d, L)

    assert len(flm_2d_check.shape) == 2
    np.testing.assert_allclose(flm_2d, flm_2d_check, atol=1e-14)


def test_flm_reindexing_exceptions(flm_generator):
    L = 16
    spin = 0

    flm_2d = flm_generator(L=L, spin=spin, reality=False)
    flm_1d = s2ball.utils.flm_2d_to_1d(flm_2d, L)
    flm_3d = np.zeros((1, 1, 1))

    with pytest.raises(ValueError) as e:
        s2ball.utils.flm_2d_to_1d(flm_1d, L)

    with pytest.raises(ValueError) as e:
        s2ball.utils.flm_2d_to_1d(flm_3d, L)

    with pytest.raises(ValueError) as e:
        s2ball.utils.flm_1d_to_2d(flm_2d, L)

    with pytest.raises(ValueError) as e:
        s2ball.utils.flm_1d_to_2d(flm_3d, L)
