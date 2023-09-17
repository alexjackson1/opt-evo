if __name__ == "__main__":
    import argparse

    import csv
    import os
    from tqdm import tqdm

    import jax.numpy as jnp
    import jax.random as jr

    import evo
    from mnist import mnist
    from models import BatchLeNet
    from loader import DataLoader

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_gen", type=int, default=10)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--crossover_p", type=float, default=0.5)
    parser.add_argument("--tournament_size", type=int, default=5)
    parser.add_argument("--elites", type=int, default=1)
    parser.add_argument("--mutation_p", type=float, default=0.01)
    parser.add_argument("--mutation_sigma", type=float, default=0.01)
    parser.add_argument("--fitness_fn", type=str, default="acc")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Configure randomness
    rng = jr.PRNGKey(args.seed)
    pop_rng, data_rng, evolve_rng = jr.split(rng, 3)

    # Initialise model and data
    model = BatchLeNet()

    train_images, train_labels, test_images, test_labels = mnist(data_dir=args.data_dir)
    loader = DataLoader(train_images, train_labels, batch_size=args.batch_size)

    # Initialise population
    ex_input = jnp.ones((1, 28, 28, 1))
    pop = evo.init_population(jr.split(pop_rng, args.pop_size), model, ex_input)

    # Set best fitness starting value
    best_individual = None
    if args.fitness_fn == "acc":
        best_fitness = 0
    elif args.fitness_fn == "loss":
        best_fitness = float("inf")
    else:
        raise ValueError(f"Unknown fitness function {args.fitness_fn}")

    results_path = os.path.join(args.out_dir, "", "results.csv")
    if not os.path.exists(results_path):
        with open(results_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "seed",
                    "max_gen",
                    "pop_size",
                    "crossover_p",
                    "tournament_size",
                    "elites",
                    "mutation_p",
                    "mutation_sigma",
                    "fitness_fn",
                    "batch_size",
                    "best_fitness",
                ]
            )

    pb = tqdm()
    for g in range(args.max_gen + 1):
        pb.set_description(f"Generation {g}")
        next_rng = jr.fold_in(evolve_rng, g)
        pop, fitness = evo.evolve(
            next_rng,
            model,
            pop,
            loader,
            elites=args.elites,
            mutation_p=args.mutation_p,
            mutation_sigma=args.mutation_sigma,
            crossover_p=args.crossover_p,
            tournament_size=args.tournament_size,
            fitness_fn=args.fitness_fn,
            progress=pb,
        )

        best_idx = jnp.argmax(fitness)
        if args.fitness_fn == "acc" and fitness[best_idx] > best_fitness:
            best_fitness = fitness[best_idx]
            best_individual = pop[best_idx]
        elif args.fitness_fn == "loss" and fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_individual = pop[best_idx]

    pb.close()

    print(f"Best fitness: {best_fitness}")

    # Write results to file
    with open(results_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                args.seed,
                args.max_gen,
                args.pop_size,
                args.crossover_p,
                args.tournament_size,
                args.elites,
                args.mutation_p,
                args.mutation_sigma,
                args.fitness_fn,
                args.batch_size,
                best_fitness,
            ]
        )

    genome_path = os.path.join(args.out_dir, "", f"model_{args.seed}.npy")
    jnp.save(genome_path, best_individual)
