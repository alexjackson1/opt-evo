import itertools
import sys
import jax
import jax.numpy as jnp

from models import BatchLeNet
from loader import DataLoader
import csv

if __name__ == "__main__":
    from mnist import mnist
    import jax.random as jr
    import evo
    from tqdm import tqdm

    rng = jr.PRNGKey(0)

    num_generations = 100
    pop_size = 300
    crossover_p = 0.5
    tournament_size = 20

    elites = [20]
    mutation_p = [0.005]
    mutation_sigma = [0.01]

    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Elites",
                "Mutation Rate",
                "Mutation Scale",
                "Best Fitness",
                "Mean Fitness",
                "Fitness STD",
                "Min Fitness",
                "Max Fitness",
            ]
        )

    for e, p, s in itertools.product(elites, mutation_p, mutation_sigma):
        print(f"Elites: {e}, Mutation p: {p}, Mutation sigma: {s}")

        pop_rng, data_rng, evolve_rng = jr.split(rng, 3)
        model = BatchLeNet()

        train_images, train_labels, test_images, test_labels = mnist()
        loader = DataLoader(train_images, train_labels, batch_size=256)

        ex_input = jnp.ones((1, 28, 28, 1))
        pop = evo.init_population(jr.split(pop_rng, pop_size), model, ex_input)

        pb = tqdm()
        for g in range(num_generations + 1):
            pb.set_description(f"Generation {g}")
            next_rng = jr.fold_in(evolve_rng, g)
            pop, fitness = evo.evolve(
                next_rng,
                model,
                pop,
                loader,
                elites=e,
                mutation_p=p,
                mutation_sigma=s,
                crossover_p=crossover_p,
                tournament_size=tournament_size,
                maximize=True,
                progress=pb,
            )

        with open("results.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    e,
                    p,
                    s,
                    fitness[0],
                    jnp.mean(fitness),
                    jnp.std(fitness),
                    jnp.min(fitness),
                    jnp.max(fitness),
                ]
            )
