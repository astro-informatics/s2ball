"""Collection of shared fixtures"""
from functools import partial
import pytest


DEFAULT_SEED = 8966433580120847635


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        nargs="*",
        default=[DEFAULT_SEED],
        help=(
            "Seed(s) to use for random number generator fixture rng in tests. If "
            "multiple seeds are passed tests depending on rng will be run for all "
            "seeds specified."
        ),
    )


def pytest_generate_tests(metafunc):
    if "seed" in metafunc.fixturenames:
        metafunc.parametrize("seed", metafunc.config.getoption("seed"))


@pytest.fixture
def rng(seed):
    import numpy as np

    return np.random.default_rng(seed)


@pytest.fixture
def flm_generator(rng):
    from s2ball import utils

    return partial(utils.generate_flm, rng)


@pytest.fixture
def flmn_generator(rng):
    from s2ball import utils

    return partial(utils.generate_flmn, rng)


@pytest.fixture
def flmp_generator(rng):
    from s2ball import utils

    return partial(utils.generate_flmp, rng)


@pytest.fixture
def flmnp_generator(rng):
    from s2ball import utils

    return partial(utils.generate_flmnp, rng)


@pytest.fixture
def s2fft_to_so3_sampling():
    def so3_sampling(s2fft_sampling):
        if s2fft_sampling.lower() == "mw":
            so3_sampling = "SO3_SAMPLING_MW"
        elif s2fft_sampling.lower() == "mwss":
            so3_sampling = "SO3_SAMPLING_MWSS"
        else:
            raise ValueError(
                f"Sampling scheme sampling={s2fft_sampling} not supported by so3."
            )

        return so3_sampling

    return so3_sampling
