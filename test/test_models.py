import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import jax.numpy as jnp

from models import Module


NUM_INPUTS = 28 * 28 * 1
FEATURES = [300, 100, 100]
NUM_CLASSES = 10


@pytest.fixture
def model_with_bias():
    return Module(features=FEATURES, num_classes=NUM_CLASSES, use_bias=True)


@pytest.fixture
def model_without_bias():
    return Module(features=FEATURES, num_classes=NUM_CLASSES, use_bias=False)


def test_num_params_without_bias(model_without_bias: Module):
    result = model_without_bias.num_params(jnp.ones((28, 28, 1)))
    assert result == 276200


def test_num_params_with_bias(model_with_bias: Module):
    result = model_with_bias.num_params(jnp.ones((28, 28, 1)))
    assert result == 276710
