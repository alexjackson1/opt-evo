from typing import Any, Dict, Generator, Literal, Optional, Tuple, Union
from jaxtyping import Array, Float

import tqdm
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

import flax.linen as nn

from models import Module
from loader import DataLoader


Params = Dict[str, Union[Array, "Params"]]
Population = Float[Array, "ind param"]
Individual = Float[Array, "param"]


def init_individual(rng: jr.KeyArray, model: Module, ex_input: Array) -> Individual:
    """
    Initialise an individual with random parameters.

    Args:
        rng: Keys to use for random number generation.
        model: The individual's phenotype (or the model architecture).
        ex_input: An example input to the model.

    Returns:
        An individual genome (or flattened model parameters)

    """
    variables = model.init(rng, ex_input)
    flattened, _ = jax.tree_flatten(variables["params"])
    params = jnp.concatenate([jnp.ravel(x) for x in flattened])
    return params


def init_population(rngs: jr.KeyArray, model: Module, ex_input: Array) -> Population:
    """
    Initialise a population of individuals.

    Args:
        rng: Tuple of keys to use for random number generation.
        model: The individual's phenotype (or the model architecture).
        ex_input: An example input to the model.

    Returns:
        A population of individuals (or flattened model parameters)
        in the shape (pop_size, num_params).
    """
    return jax.vmap(init_individual, in_axes=(0, None, None))(rngs, model, ex_input)


def to_variables(model: Module, ind: Individual, input_size: int) -> Params:
    """
    Convert an individual's genome into a dictionary of model parameters.

    Args:
        model: The individual's phenotype (or the model architecture).
        ind: The individual's genome (or flattened model parameters).
        input_size: The size of the input to the model.

    Returns:
        A dictionary of model parameters.
    """
    features = [input_size] + model.features + [model.num_classes]
    params, position = {}, 0
    for i in range(len(features) - 1):
        # Calculate flattened size
        d1, d2 = features[i], features[i + 1]
        size = int(d1 * d2)

        # Reshape to parameter dimensions
        subset = ind[position : position + size]
        weight = jnp.reshape(subset, (d1, d2))
        position += size
        params[f"Dense_{i}"] = {"kernel": weight}

        # Extract bias if using
        if model.use_bias:
            bias = ind[position : position + d2]
            position += d2
            params[f"Dense_{i}"]["bias"] = bias

    # Return parameters
    return {"params": params}


def mutate_individual(
    rng: jr.KeyArray,
    ind: Individual,
    p: float = 0.1,
    sigma: float = 0.1,
):
    """
    Mutate an individual's genome.

    Args:
        rng: Keys to use for random number generation.
        ind: The individual's genome (or flattened model parameters).
        p: The probability of mutating each gene.
        sigma: The standard deviation of the noise to add to each gene.

    Returns:
        The mutated individual's genome.
    """
    noise = jr.normal(rng, ind.shape) * sigma
    mask = jr.bernoulli(rng, p, ind.shape)
    return ind + noise * mask


def mutate_population(
    rngs: jr.KeyArray,
    pop: Population,
    p: float = 0.1,
    sigma: float = 0.1,
):
    """
    Mutate a population of individuals.

    Args:
        rng: Tuple of keys to use for random number generation.
        pop: The population of individuals (or flattened model parameters).
        p: The probability of mutating each gene.
        sigma: The standard deviation of the noise to add to each gene.

    Returns:
        The mutated population of individuals.
    """

    def mutate(rng: jr.KeyArray, individual: Individual) -> Individual:
        return mutate_individual(rng, individual, p, sigma)

    return jax.vmap(mutate, in_axes=(0, 0))(rngs, pop)


def tournament_selection(
    rng: jr.KeyArray,
    pop: Population,
    fitness: Float[Array, "ind"],
    size: int,
    maximize: bool = True,
) -> Generator[Tuple[int, int], Any, None]:
    """
    Select two individuals from a population using tournament selection.

    Args:
        rng: Keys to use for random number generation.
        pop: The population of individuals (or flattened model parameters).
        fitness: The fitness of each individual.
        size: The number of individuals to sample for each tournament.
        maximize: Whether to maximise or minimise the fitness.

    Yields:
        A tuple of the indices of the two selected individuals.
    """

    num_individuals = pop.shape[0]

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


def crossover_population(
    rng: jr.KeyArray,
    pop: Population,
    fitness: Float[Array, "ind"],
    num_offspring: int = 1,
    p: float = 0.5,
    tournament_size: int = 5,
    maximize: bool = True,
) -> Population:
    """
    Crossover a population of individuals.

    Args:
        rng: Keys to use for random number generation.
        pop: The population of individuals (or flattened model parameters).
        fitness: The fitness of each individual.
        elites: The number of elite individuals to keep in the population.
        p: The probability of crossing over each gene.
        tournament_size: The number of individuals to sample for each tournament.
        maximize: Whether to maximise or minimise the fitness.

    Returns:
        The new population of individuals.
    """

    selection_key, crossover_key = jr.split(rng)

    # Select the parents for each offspring using tournament selection
    parents = tournament_selection(
        selection_key,
        pop,
        fitness,
        size=tournament_size,
        maximize=maximize,
    )

    children = []
    iters = int(num_offspring // 2 + num_offspring % 2)
    for i in range(iters):
        # Select the parents for the next offspring
        parent_1_idx, parent_2_idx = next(parents)
        parent_1, parent_2 = pop[parent_1_idx], pop[parent_2_idx]

        # Crossover the parents to create the next offspring
        mask = jr.bernoulli(jr.fold_in(crossover_key, i), p, parent_1.shape)
        child_1 = jnp.where(mask, parent_1, parent_2)
        children.append(child_1)

        # Only create the second child if necessary
        if len(children) < num_offspring:
            child_2 = jnp.where(mask, parent_2, parent_1)
            children.append(child_2)

    assert len(children) == num_offspring

    return jnp.stack(children)


def evaluate_accuracy(
    model: Module,
    vars: Params,
    ldr: DataLoader,
    progress: Optional[tqdm.tqdm],
) -> float:
    """
    Evaluate the accuracy of a model on a dataset.

    Args:
        model: The model to evaluate.
        vars: The model's parameters.
        ldr: The dataset to evaluate on.
        progress: A tqdm progress bar to update.

    Returns:
        The accuracy of the model on the dataset.
    """
    num_correct = 0
    total = 0

    for x, y in ldr:
        if progress is not None:
            progress.update(1)

        y_hat = model.apply(vars, x)
        y_hat = jnp.argmax(y_hat, axis=1)
        num_correct += jnp.sum(y_hat == y)
        total += y.shape[0]

    return num_correct / total


def evaluate_cross_entropy_loss(
    model: Module,
    vars: Params,
    ldr: DataLoader,
    progress: Optional[tqdm.tqdm],
) -> float:
    """
    Evaluate the cross entropy loss of a model on a dataset.

    Args:
        model: The model to evaluate.
        vars: The model's parameters.
        ldr: The dataset to evaluate on.
        progress: A tqdm progress bar to update.

    Returns:
        The cross entropy loss of the model on the dataset.
    """
    loss = 0.0
    total = 0

    for x, y in ldr:
        if progress is not None:
            progress.update(1)

        y_hat = model.apply(vars, x)
        loss += jnp.sum(nn.log_softmax(y_hat) * nn.one_hot(y, 10))
        total += y.shape[0]

    return -loss / total


def evolve(
    rng: jr.KeyArray,
    model: Module,
    pop: Float[Array, "ind param"],
    ldr: DataLoader,
    elites: int = 0,
    mutation_p: float = 0.05,
    mutation_sigma: float = 0.1,
    crossover_p: float = 0.5,
    tournament_size: int = 5,
    fitness_fn: Literal["acc", "loss"] = "acc",
    progress: Optional[tqdm.tqdm] = None,
) -> Float[Array, "ind param"]:
    """
    Evolve a population of individuals.

    Args:
        rng: Keys to use for random number generation.
        model: The individual's phenotype (or the model architecture).
        pop: The population of individuals (or flattened model parameters).
        ldr: The dataset to evaluate on.
        elites: The number of elite individuals to keep in the population.
        mutation_p: The probability of mutating each gene.
        mutation_sigma: The standard deviation of the noise to add to each gene.
        crossover_p: The probability of crossing over each gene.
        tournament_size: The number of individuals to sample for each tournament.
        fitness_fn: The fitness function to use.
        progress: A tqdm progress bar to update.

    Returns:
        The new population of individuals.
    """
    crossover_key, mutation_key = jr.split(rng)

    if fitness_fn == "acc":
        maximize = True
        calc_fitness = evaluate_accuracy
    elif fitness_fn == "loss":
        maximize = False
        calc_fitness = evaluate_cross_entropy_loss
    else:
        raise ValueError(f"Unknown fitness function: {fitness_fn}")

    def fitness_fn(individual: Float[Array, "param"]) -> float:
        input_size = np.prod(ldr[0].x.shape)
        variables = to_variables(model, individual, input_size)
        return calc_fitness(model, variables, ldr, progress)

    # Calculate the fitness of each individual
    fitness: Float[Array, "ind"] = jax.vmap(fitness_fn)(pop)

    if progress is not None:
        if maximize:
            progress.set_postfix(
                {
                    "max": np.max(fitness),
                    "mean": np.mean(fitness),
                    "count": len(fitness),
                }
            )
        else:
            progress.set_postfix(
                {
                    "min": np.min(fitness),
                    "mean": np.mean(fitness),
                    "count": len(fitness),
                }
            )

    # Select the elite individuals
    if elites:
        if maximize:
            elite_idxs = jnp.argsort(fitness)[-elites:]
        else:
            elite_idxs = jnp.argsort(fitness)[:elites]

        elite_pop = pop[elite_idxs]

    # Create the offspring
    num_offspring = pop.shape[0] - elites
    children = crossover_population(
        crossover_key,
        pop,
        fitness,
        num_offspring,
        p=crossover_p,
        tournament_size=tournament_size,
        maximize=maximize,
    )

    # Mutate the population (excluding the elites)
    mutation_keys = jr.split(mutation_key, int(num_offspring))
    children = mutate_population(
        mutation_keys, children, p=mutation_p, sigma=mutation_sigma
    )

    # Combine the elites and the offspring
    if elites:
        final_pop = jnp.concatenate([elite_pop, children])
    else:
        final_pop = children

    return final_pop.block_until_ready(), fitness.block_until_ready()
