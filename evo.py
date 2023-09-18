from dataclasses import dataclass
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union
from jaxtyping import Array, Float, Int, PyTreeDef

import jax
import jax.numpy as jnp
import jax.random as jr

import flax.linen as nn


class Population(NamedTuple):
    genes: Float[Array, "P Θ"]
    shapes: List[Tuple[int, ...]]
    tree_def: PyTreeDef


def init_population(
    rngs: jr.KeyArray, model: nn.Module, input_shape: Array
) -> Population:
    """
    Initialise a population of individuals.

    Args:
        rng: Keys to use for random number generation.
        model: The model to initialise.
        input_shape: The shape of the input to the model.

    Returns:
        A tuple of the population of individuals and the tree definition of the
        model.
    """
    dummy_input = jnp.ones(input_shape)
    leafs, tree_def = jax.tree_flatten(model.init(jr.PRNGKey(0), dummy_input)["params"])
    shapes = [leaf.shape for leaf in leafs]

    def init_individual(rng: jr.KeyArray) -> Float[Array, "Θ"]:
        params = model.init(rng, dummy_input)["params"]
        leafs, _ = jax.tree_flatten(params)
        return jnp.concatenate([leaf.ravel() for leaf in leafs])

    genes = jax.vmap(init_individual)(rngs)
    return Population(genes, shapes, tree_def)


def tournament_selection(
    rng: jr.KeyArray,
    fitness: Float[Array, "P"],
    size: int,
    maximize: bool = True,
) -> Generator[Tuple[int, int], Any, None]:
    """
    Select two individuals from a population using tournament selection.

    Args:
        rng: Keys to use for random number generation.
        fitness: The fitness of each individual.
        size: The number of individuals to sample for each tournament.
        maximize: Whether to maximise or minimise the fitness.

    Yields:
        A tuple of the indices of the two selected individuals.
    """

    num_individuals = fitness.shape[0]

    if num_individuals < 2:
        raise ValueError("Population size must be at least 2")

    iterations = 0
    while True:
        iterations += 1

        # Randomly sample indices for the tournament
        key = jr.fold_in(rng, iterations)
        indicies = jr.randint(key, (size,), 0, num_individuals)

        # Select the individuals with the two highest fitness values from the tournament
        if maximize:
            winner_indices = jnp.argsort(fitness[indicies])[-2:]
        else:
            winner_indices = jnp.argsort(fitness[indicies])[:2]

        yield indicies[winner_indices[0]], indicies[winner_indices[1]]


def crossover(
    rng: jr.KeyArray,
    parents: Tuple[Float[Array, "Θ"], Float[Array, "Θ"]],
    prob: float = 0.5,
    offspring: int = 2,
):
    """
    Crossover two individuals.

    Args:
        rng: Keys to use for random number generation.
        parents: The two individuals to crossover.
        prob: The probability of crossing over each gene.
        offspring: The number of offspring to create.

    Returns:
        The offspring.
    """
    if offspring not in [1, 2]:
        raise ValueError("Number of offspring must be 1 or 2")

    parent_1, parent_2 = parents

    # Crossover the parents to create the next offspring
    mask = jr.bernoulli(rng, prob, parent_1.shape)
    child_1 = jnp.where(mask, parent_1, parent_2)

    # If only one offspring is required, return it
    if offspring == 1:
        return child_1

    # Otherwise, create the second offspring
    child_2 = jnp.where(mask, parent_2, parent_1)
    return child_1, child_2


def add_noise(rng: jr.KeyArray, a: Array, prob: float = 0.1, sigma: float = 0.1):
    """
    Add noise to an array with arbitrary shape.

    Args:
        rng: Keys to use for random number generation.
        a: The array to add noise to.
        prob: The probability of mutating each gene.
        sigma: The standard deviation of the noise to add to each gene.

    Returns:
        The array with noise added.
    """
    mask = jr.bernoulli(rng, prob, a.shape)
    return a + jr.normal(rng, a.shape) * sigma * mask


def find_elites(
    fitness: Float[Array, "P Θ"], elites: int, maximize: bool
) -> Optional[Int[Array, "E"]]:
    """
    Identify the elite individuals in a population.

    Args:
        fitness: The fitness of each individual.
        elites: The number of elite individuals to keep in the population.
        maximize: Whether to maximise or minimise the fitness.

    Returns:
        The elite individual indices.
    """
    if elites is None or elites <= 0:
        return None

    if elites > fitness.shape[0] - 1:
        raise ValueError("Number of elites must be less than the population size")

    sorted_args = jnp.argsort(fitness)
    if maximize:
        elite_idxs = sorted_args[-elites:]
    else:
        elite_idxs = sorted_args[:elites]

    return elite_idxs
