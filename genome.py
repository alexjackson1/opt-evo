from dataclasses import dataclass
from typing import Tuple, TypedDict
from jaxtyping import Array, Float, PyTreeDef

import jax
import jax.numpy as jnp
import jax.random as jr

import flax.linen as nn
import flax.struct as struct
from optax import Params


@dataclass
class Genome:
    genes: Float[Array, "θ"] = struct.field(pytree_node=True)
    structure: PyTreeDef = struct.field(pytree_node=False)

    def __init__(self, genes: Float[Array, "θ"], structure: PyTreeDef):
        self.genes = genes
        self.structure = structure

    @staticmethod
    def create(params: Params) -> "Genome":
        leaves, structure = jax.tree_flatten(params)[0]
        genes = jnp.concatenate([jnp.ravel(x) for x in leaves])
        return Genome(genes, structure)

    def update(self, params: Params) -> "Genome":
        return Genome.create(params)

    def phenotype(self) -> Params:
        return jax.tree_unflatten(self.structure, self.genes)

    @property
    def num_params(self) -> int:
        return self.genes.shape[0]

    @property
    def num_leaves(self) -> int:
        return len(self.structure)

    def __repr__(self):
        return f"Genome(num_params={self.num_params}, num_leaves={self.num_leaves})"
