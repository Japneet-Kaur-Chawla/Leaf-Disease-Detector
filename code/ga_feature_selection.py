# ga_feature_selection.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def chromosome_fitness(X, y, chromosome):
    selected = np.where(np.array(chromosome) == 1)[0]

    if len(selected) == 0:
        return 0.0

    X_selected = X[:, selected]

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  

    scores = cross_val_score(
        clf,
        X_selected,
        y,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1
    )

    accuracy = scores.mean()

    feature_ratio = len(selected) / X.shape[1]

    return 0.95 * accuracy + 0.05 * (1 - feature_ratio)

def run_ga(X, y, pop_size=40, n_gens=25, mutation_rate=0.02, crossover_rate=0.8):

    n_features = X.shape[1]

    # Better sparse initialization
    population = [
        np.random.choice(
            [0, 1],
            size=n_features,
            p=[0.7, 0.3]
        ).tolist()
        for _ in range(pop_size)
    ]

    best_chromosome = None
    best_fitness = 0.0

    print("[GA] Running Genetic Algorithm...")

    for gen in range(n_gens):

        fitness_scores = []

        for chrom in population:

            fitness = chromosome_fitness(X, y, chrom)

            fitness_scores.append(fitness)

        gen_best_idx = int(np.argmax(fitness_scores))

        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness = fitness_scores[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()

        print(
            f"[GA] Generation {gen+1}/{n_gens} "
            f"Best Fitness={best_fitness:.4f} "
            f"Selected Features={np.sum(best_chromosome)}"
        )

        # Elitism
        new_population = [population[gen_best_idx]]

        while len(new_population) < pop_size:

            parents = np.random.choice(
                pop_size,
                size=2,
                replace=False
            )

            p1 = population[parents[0]]
            p2 = population[parents[1]]

            # Crossover
            if np.random.rand() < crossover_rate:

                point = np.random.randint(1, n_features)

                child1 = p1[:point] + p2[point:]
                child2 = p2[:point] + p1[point:]

            else:
                child1 = p1.copy()
                child2 = p2.copy()

            # Mutation
            for child in [child1, child2]:

                for i in range(n_features):

                    if np.random.rand() < mutation_rate:
                        child[i] = 1 - child[i]

                # Ensure at least one feature
                if np.sum(child) == 0:
                    child[np.random.randint(n_features)] = 1

                new_population.append(child)

                if len(new_population) >= pop_size:
                    break

        population = new_population[:pop_size]

    print("[GA] Finished!")

    return np.array(best_chromosome, dtype=np.int8)


def apply_feature_mask(npz_file, out_file, mask):

    d = np.load(npz_file)

    X = d["X"]
    y = d["y"]

    selected = mask.astype(bool)

    np.savez(
        out_file,
        X=X[:, selected],
        y=y
    )