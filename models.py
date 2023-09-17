from typing import List
from jaxtyping import Array, Float

import jax.numpy as jnp
import flax.linen as nn
import flax.struct as struct

from loader import DataLoader


class Module(nn.Module):
    features: List[int] = struct.field(default_factory=lambda: [300, 100, 100])
    num_classes: int = 10
    use_bias: bool = True

    def num_params(self, ex_input: Array) -> int:
        input_size = jnp.prod(jnp.array(ex_input.shape))
        layers = [input_size] + self.features + [self.num_classes]
        weights = sum([d1 * d2 for d1, d2 in zip(layers[:-1], layers[1:])])
        if self.use_bias:
            biases = sum([d2 for d2 in layers[1:]])
        else:
            biases = 0
        return weights + biases


class LeNet(Module):
    @nn.compact
    def __call__(self, x: Float[Array, "x1 x2 ch"]) -> Float[Array, "cl"]:
        x = jnp.reshape(x, -1)  # flatten

        # apply layers
        for num_features in self.features:
            x = nn.Dense(num_features, use_bias=self.use_bias)(x)
            x = nn.relu(x)

        x = nn.Dense(self.num_classes, use_bias=self.use_bias)(x)
        x = nn.softmax(x, axis=0)  # normalise
        return x


BatchLeNet = nn.vmap(
    LeNet,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None},
    split_rngs={"params": False},
)
