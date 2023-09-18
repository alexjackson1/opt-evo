import argparse
from typing import Callable

import flax.linen as nn
import numpy as np
from models import BatchConv, BatchFeedForward


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--arch", type=str, default="conv")

    # GA parameters
    parser.add_argument("--max_gen", type=int, default=10)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--elites", type=int, default=1)
    parser.add_argument("--crossover_p", type=float, default=0.5)
    parser.add_argument("--tournament_size", type=int, default=5)
    parser.add_argument("--noise_p", type=float, default=0.01)
    parser.add_argument("--noise_sigma", type=float, default=0.01)
    parser.add_argument("--criterion", type=str, default="acc")
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()
    return args


def _parse_arch(arch: str) -> Callable[..., nn.Module]:
    if arch == "conv":
        return BatchConv
    elif arch == "ff":
        return BatchFeedForward
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def _make_results_path(args: argparse.Namespace) -> str:
    path_components = [
        "results" f"seed_{args.seed}",
        f"arch_{args.arch}",
        f"pop_{args.pop_size}",
        f"criterion_{args.criterion}",
        f"elites_{args.elites}",
        f"tournament_{args.tournament_size}",
        f"noise_{args.noise_p}_{args.noise_sigma}",
        f"batch_{args.batch_size}",
    ]
    return os.path.join(args.out_dir, "", "_".join(path_components) + ".csv")


if __name__ == "__main__":
    from jaxtyping import Array, Float

    import argparse
    import csv
    import os

    import jax.numpy as jnp
    import jax.random as jr

    from mnist import mnist
    from models import BatchFeedForward, BatchConv
    from loader import DataLoader
    from ga import EvolveState, GAConfig, ga

    # parse arguments
    args = cli()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Configure randomness
    rng = jr.PRNGKey(args.seed)

    # Initialise model
    arch = _parse_arch(args.arch)
    model = arch(features=[150, 50, 50])

    # Initialise dataset
    train_images, train_labels, test_images, test_labels = mnist(data_dir=args.data_dir)
    loader = DataLoader(train_images, train_labels, batch_size=args.batch_size)

    # Initialise callbacks
    results_path = _make_results_path(args)

    def initialise_callback() -> EvolveState:
        return EvolveState()

    def generation_callback(
        state: EvolveState,
        population: Float[Array, "P Θ"],
        fitness: Float[Array, "P"],
        maximise: bool = True,
    ) -> EvolveState:
        # Check if best individual has improved
        if maximise:
            best_idx = jnp.argmax(fitness)
            if np.isnan(state.best_fitness) or fitness[best_idx] > state.best_fitness:
                best_fitness = fitness[best_idx]
                best_individual = population[best_idx]
            else:
                best_fitness = state.best_fitness
                best_individual = state.best_individual
        else:
            best_idx = jnp.argmin(fitness)
            if np.isnan(state.best_fitness) or fitness[best_idx] < state.best_fitness:
                best_fitness = fitness[best_idx]
                best_individual = population[best_idx]
            else:
                best_fitness = state.best_fitness
                best_individual = state.best_individual

        # Update results
        return EvolveState(
            generation=state.generation + 1,
            best_fitness=best_fitness,
            best_individual=best_individual,
        )

    config = GAConfig(
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        num_elites=args.elites,
        criterion=args.criterion,
        prob_add_noise=args.noise_p,
        prob_crossover=args.crossover_p,
        noise_sigma=args.noise_sigma,
        tournament_size=args.tournament_size,
        batch_size=args.batch_size,
    )

    state = ga(rng, model, loader, initialise_callback, generation_callback, config)

    print(f"Best fitness: {state.best_fitness}")
    # print(f"Best individual: {state.best_individual}")
