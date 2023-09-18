import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import evo
from models import FeedForward, Module
import jax
import jax.numpy as jnp
import jax.random as jr


@pytest.fixture
def model():
    return FeedForward(features=[300, 100, 100], num_classes=10, use_bias=True)


def test_init_individual(model: Module):
    rng = jr.PRNGKey(0)
    ex_input = jnp.ones((1, 28, 28, 1))
    params = evo.init_individual(rng, model, ex_input)
    assert params.shape == (model.num_params(ex_input),)


def test_init_population(model: Module):
    rng = jr.PRNGKey(0)
    pop_size = 10
    ex_input = jnp.ones((1, 28, 28, 1))
    num_params = model.num_params(ex_input)
    params = evo.init_population(jr.split(rng, pop_size), model, ex_input)
    assert params.shape == (pop_size, num_params)
    assert not jnp.allclose(params[0], params[1], atol=1e-3, rtol=1e-3)
