import csv
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Literal, Tuple, TypeVar
from jaxtyping import Array, Float, Int, PyTreeDef

import tqdm
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

import flax.linen as nn
from optax import Params

import evo
from loader import DataLoader


FitnessFn = Callable[[nn.Module, Params, DataLoader, Optional[tqdm.tqdm]], float]


def cross_entropy_loss(logits: Float[Array, "B C"], labels: Int[Array, "B"]) -> float:
    onehot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(onehot * jax.nn.log_softmax(logits), axis=1))


def _calc_full_loss(
    model: nn.Module,
    variables: Params,
    loader: DataLoader,
    progress: Optional[tqdm.tqdm],
) -> float:
    loss = 0
    total = 0

    for x, y in loader:
        if progress is not None:
            progress.update(1)

        y_hat = model.apply(variables, x)
        loss += cross_entropy_loss(y_hat, y)
        total += y.shape[0]

    return loss / total


def _calc_batch_loss(
    model: nn.Module,
    variables: Params,
    loader: DataLoader,
    progress: Optional[tqdm.tqdm],
) -> float:
    x, y = next(loader.__iter__())
    if progress is not None:
        progress.update(1)

    y_hat = model.apply(variables, x)
    return cross_entropy_loss(y_hat, y)


def _calc_full_acc(
    model: nn.Module,
    vars: Params,
    ldr: DataLoader,
    progress: Optional[tqdm.tqdm],
) -> float:
    num_correct = 0
    total = 0

    for x, y in ldr:
        if progress is not None:
            progress.update(1)

        y_hat = model.apply(vars, x)
        labels = jnp.argmax(y_hat, axis=1)
        num_correct += jnp.sum(labels == y)
        total += y.shape[0]

    return num_correct / total


def _calc_batch_acc(
    model: nn.Module,
    vars: Params,
    ldr: DataLoader,
    progress: Optional[tqdm.tqdm],
) -> float:
    x, y = next(ldr.__iter__())
    if progress is not None:
        progress.update(1)

    y_hat = model.apply(vars, x)
    labels = jnp.argmax(y_hat, axis=1)
    num_correct = jnp.sum(labels == y)
    return num_correct / y.shape[0]


def _parse_fitness_fn(
    criterion: Literal["acc", "loss", "batch_acc", "batch_loss"]
) -> FitnessFn:
    if criterion == "acc":
        return _calc_full_acc
    elif criterion == "loss":
        return _calc_full_loss
    elif criterion == "batch_acc":
        return _calc_batch_acc
    elif criterion == "batch_loss":
        return _calc_batch_loss
    else:
        raise ValueError(f"Unknown fitness function {criterion}")


def _update_progress(progress: Optional[tqdm.tqdm], fitness: Float[Array, "i"]):
    if progress is not None:
        stats = {
            "max": np.max(fitness),
            "min": np.min(fitness),
            "mean": np.mean(fitness),
            "std": np.std(fitness),
        }
        progress.set_postfix(stats)


def to_phenotype(
    individual: Float[Array, "Θ"], shapes: List[Tuple[int, ...]], tree_def: PyTreeDef
) -> Params:
    """
    Convert an individual's genotype to its phenotype.

    Args:
        individual: The individual's genotype.
        shapes: The shapes of the individual's genes.
        tree_def: The tree definition of the individual's genes.

    Returns:
        The individual's phenotype.
    """
    layers, start_idx = [], 0
    for shape in shapes:
        size = np.prod(shape)
        end_idx = start_idx + size
        layers.append(individual[start_idx:end_idx].reshape(shape))
        start_idx += size

    return jax.tree_unflatten(tree_def, layers)


def evolve(
    rng: jr.KeyArray,
    model: nn.Module,
    population: evo.Population,
    loader: DataLoader,
    criterion: Literal["acc", "loss"] = "acc",
    num_elites: int = 0,
    prob_add_noise: float = 0.05,
    prob_crossover: float = 0.5,
    noise_sigma: float = 0.1,
    tournament_size: int = 5,
    progress: Optional[tqdm.tqdm] = None,
) -> Float[Array, "ind param"]:
    """
    Evolve a population of individuals.

    Args:
        rng: Keys to use for random number generation.
        model: The individual's phenotype (or the model architecture).
        population: The population of individuals (or flattened model parameters).
        loader: The dataset to evaluate on.
        criterion: The fitness function to use.
        num_elites: The number of elite individuals to keep in the population.
        prob_add_noise: The probability of mutating each gene.
        prob_crossover: The probability of crossing over each gene.
        noise_sigma: The standard deviation of the noise to add to each gene.
        tournament_size: The number of individuals to sample for each tournament.
        progress: A tqdm progress bar to update.

    Returns:
        The new population of individuals.
    """
    selection_key, crossover_key, mutation_key = jr.split(rng, 3)

    # Parse the fitness function
    calc_fitness = _parse_fitness_fn(criterion)
    maximize = criterion in ["acc", "batch_acc"]

    def fitness_fn(individual: Float[Array, "Θ"]) -> float:
        params = to_phenotype(individual, population.shapes, population.tree_def)
        variables = {"params": params}
        return calc_fitness(model, variables, loader, progress)

    # Compute fitness
    fitness = jax.vmap(fitness_fn)(population.genes)
    _update_progress(progress, fitness)

    # Select the elite individuals
    elite_idxs = evo.find_elites(fitness, num_elites, maximize=maximize)
    if elite_idxs is not None:
        elite_pop = population.genes[elite_idxs]

    # Select the parents for each offspring using tournament selection
    parent_gen = evo.tournament_selection(
        selection_key, fitness, tournament_size, maximize
    )

    # Create the offspring
    children = []
    for _ in range(population.genes.shape[0] - num_elites):
        parent_1_idx, parent_2_idx = next(parent_gen)
        parents = population.genes[parent_1_idx], population.genes[parent_2_idx]
        child = evo.crossover(crossover_key, parents, prob_crossover, offspring=1)
        children.append(child)

    offspring = jnp.stack(children)

    # Mutate the population (excluding the elites)
    offspring = evo.add_noise(mutation_key, offspring, prob_add_noise, noise_sigma)

    # Combine the elites and the offspring
    if elite_idxs is not None:
        final_pop = jnp.concatenate([elite_pop, offspring])
    else:
        final_pop = offspring

    return final_pop.block_until_ready(), fitness.block_until_ready()


@dataclass
class GAConfig:
    pop_size: int = 100
    max_gen: int = 100
    num_elites: int = 1
    criterion: Literal["acc", "loss", "batch_acc", "batch_loss"] = "acc"
    prob_add_noise: float = 0.05
    prob_crossover: float = 0.5
    noise_sigma: float = 0.1
    tournament_size: int = 5
    batch_size: int = 32


@dataclass
class EvolveState:
    generation: int = 0
    best_fitness: float = np.nan
    best_individual: Float[Array, "Θ"] = None


State = TypeVar("State")


def ga(
    rng: jr.KeyArray,
    model: nn.Module,
    loader: DataLoader,
    init_cb: Callable[[], State],
    iter_cb: Callable[[State, Float[Array, "P Θ"], Float[Array, "P"]], State],
    config: GAConfig = GAConfig(),
):
    init_rng, evolve_rng = jr.split(rng)

    # Initialise population
    init_keys = jr.split(init_rng, config.pop_size)
    population = evo.init_population(init_keys, model, (1, 28, 28, 1))
    print(population.genes.shape)
    print(population.shapes)
    pb = tqdm.tqdm()
    state = init_cb()
    for g in range(config.max_gen + 1):
        pb.set_description(f"Generation {g}")

        new_pop, fitness = evolve(
            jr.fold_in(evolve_rng, g),
            model,
            population,
            loader,
            criterion=config.criterion,
            num_elites=config.num_elites,
            prob_add_noise=config.prob_add_noise,
            noise_sigma=config.noise_sigma,
            prob_crossover=config.prob_crossover,
            tournament_size=config.tournament_size,
            progress=pb,
        )

        state = iter_cb(state, new_pop, fitness)
        population = evo.Population(new_pop, population.shapes, population.tree_def)

    pb.close()

    return state
