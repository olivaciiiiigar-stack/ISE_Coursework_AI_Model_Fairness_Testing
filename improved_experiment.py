import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import time
import csv
import multiprocessing as mp
from tqdm import tqdm
from scipy.stats import wilcoxon

# Dataset configuration (same as baseline)
datasets = [
    {
        'name': 'ADULT',
        'file': 'ise-lab-solution/lab4/dataset/processed_adult.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_adult.h5',
        'target': 'Class-label',
        'sensitive': ['gender', 'race', 'age']
    },
    {
        'name': 'COMPAS',
        'file': 'ise-lab-solution/lab4/dataset/processed_compas.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_compas.h5',
        'target': 'Recidivism',
        'sensitive': ['Sex', 'Race']
    },
    {
        'name': 'LAW SCHOOL',
        'file': 'ise-lab-solution/lab4/dataset/processed_law_school.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_law_school_cleaned.h5',
        'target': 'pass_bar',
        'sensitive': ['male', 'race']
    },
    {
        'name': 'KDD',
        'file': 'ise-lab-solution/lab4/dataset/processed_kdd.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_kdd_cleaned.h5',
        'target': 'income',
        'sensitive': ['sex', 'race']
    },
    {
        'name': 'DUTCH',
        'file': 'ise-lab-solution/lab4/dataset/processed_dutch.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_dutch.h5',
        'target': 'occupation',
        'sensitive': ['sex', 'age']
    },
    {
        'name': 'CREDIT',
        'file': 'ise-lab-solution/lab4/dataset/processed_credit_with_numerical.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_credit.h5',
        'target': 'class',
        'sensitive': ['SEX', 'EDUCATION', 'MARRIAGE']
    },
    {
        'name': 'CRIME',
        'file': 'ise-lab-solution/lab4/dataset/processed_communities_crime.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_communities_crime.h5',
        'target': 'class',
        'sensitive': ['Black', 'FemalePctDiv']
    },
    {
        'name': 'GERMAN',
        'file': 'ise-lab-solution/lab4/dataset/processed_german.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_greman_cleaned.h5',
        'target': 'CREDITRATING',
        'sensitive': ['PersonStatusSex', 'AgeInYears']
    }
]


def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


# ============================================================
# Baseline: Random Search (same logic as comprehensive_experiment)
# ============================================================
def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns):
    """Generate a single test sample pair (same as original lab4 solution)."""
    sample_a = X_test.iloc[np.random.choice(len(X_test))]
    sample_b = sample_a.copy()

    for col in sensitive_columns:
        if col in X_test.columns:
            unique_values = X_test[col].unique()
            sample_b[col] = np.random.choice(unique_values)

    for col in non_sensitive_columns:
        if col in X_test.columns:
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
            if pd.api.types.is_integer_dtype(X_test[col]):
                sample_a[col] = int(np.clip(sample_a[col] + perturbation, min_val, max_val))
                sample_b[col] = int(np.clip(sample_b[col] + perturbation, min_val, max_val))
            else:
                sample_a[col] = np.clip(sample_a[col] + perturbation, min_val, max_val)
                sample_b[col] = np.clip(sample_b[col] + perturbation, min_val, max_val)

    return sample_a, sample_b


def evaluate_discrimination(model, sample_a, sample_b, threshold=0.05):
    """Evaluate whether a sample pair is discriminatory (single prediction)."""
    sample_a = np.array(sample_a)
    sample_b = np.array(sample_b)
    prediction_a = model.predict(sample_a.reshape(1, -1), verbose=0)
    prediction_b = model.predict(sample_b.reshape(1, -1), verbose=0)
    pred_a = prediction_a[0][0]
    pred_b = prediction_b[0][0]
    if abs(pred_a - pred_b) > threshold:
        return 1
    else:
        return 0


def random_search(model, X_test, sensitive_cols, non_sensitive_cols, num_samples=1000, threshold=0.05):
    """Pure random search — baseline method (single-sample prediction)."""
    disc_count = 0
    for _ in range(num_samples):
        sample_a, sample_b = generate_sample_pair(X_test, sensitive_cols, non_sensitive_cols)
        disc_count += evaluate_discrimination(model, sample_a, sample_b, threshold)
    return disc_count, num_samples


# ============================================================
# Improved: GA Global Search + Local Search Refinement
# ============================================================

def _flip_sensitive_single(sample, X_test, sensitive_cols):
    """Create counterpart by flipping sensitive features for a single sample."""
    flipped = sample.copy()
    for col in sensitive_cols:
        if col in X_test.columns:
            flipped[col] = np.random.choice(X_test[col].unique())
    return flipped


def _evaluate_single(model, sample_a, sample_b, threshold=0.05):
    """Evaluate a single pair, return (is_disc, fitness)."""
    a = np.array(sample_a).reshape(1, -1)
    b = np.array(sample_b).reshape(1, -1)
    pred_a = model.predict(a, verbose=0)[0][0]
    pred_b = model.predict(b, verbose=0)[0][0]
    diff = abs(pred_a - pred_b)
    return (1 if diff > threshold else 0), diff


def ga_local_search(model, X_test, sensitive_cols, non_sensitive_cols,
                    num_samples=1000, threshold=0.05,
                    pop_size=200, ga_generations=5, local_budget_ratio=0.4,
                    tournament_k=3, crossover_rate=0.8, mutation_rate=0.3,
                    local_neighbours=5):
    """
    Two-phase fairness testing (single-sample prediction):
      Phase 1 — GA global search to locate discriminatory regions
      Phase 2 — Local search around discovered instances to find more

    Total model evaluations are kept at roughly `num_samples` pairs
    so the comparison with random search is fair.
    """
    columns = list(X_test.columns)
    nonsens_idx = [columns.index(c) for c in non_sensitive_cols if c in columns]
    X_vals = X_test.values
    col_min = X_vals.min(axis=0)
    col_max = X_vals.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    int_cols = set(i for i, c in enumerate(columns) if pd.api.types.is_integer_dtype(X_test[c]))

    total_evals = 0
    disc_count = 0

    ga_budget = int(num_samples * (1 - local_budget_ratio))

    # --- Phase 1: Genetic Algorithm ---
    init_idx = np.random.choice(len(X_test), size=pop_size, replace=True)
    population = X_vals[init_idx].astype(float).copy()

    max_ga_gens = max(1, ga_generations)
    if pop_size * max_ga_gens > ga_budget:
        max_ga_gens = max(1, ga_budget // pop_size)

    disc_samples_phase1 = []

    for gen in range(max_ga_gens):
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            sample_a = pd.Series(population[i], index=columns)
            sample_b = _flip_sensitive_single(sample_a, X_test, sensitive_cols)
            is_disc, fit = _evaluate_single(model, sample_a, sample_b, threshold)
            fitness[i] = fit
            total_evals += 1
            disc_count += is_disc
            if is_disc:
                disc_samples_phase1.append(population[i].copy())

        # --- Selection (tournament) ---
        new_pop = np.empty_like(population)
        for i in range(pop_size):
            candidates = np.random.choice(pop_size, size=tournament_k, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            new_pop[i] = population[winner]

        # --- Crossover (uniform on non-sensitive features) ---
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < crossover_rate:
                mask = np.random.rand(len(nonsens_idx)) < 0.5
                for j, ci in enumerate(nonsens_idx):
                    if mask[j]:
                        new_pop[i, ci], new_pop[i + 1, ci] = new_pop[i + 1, ci], new_pop[i, ci]

        # --- Mutation (small perturbation on non-sensitive features) ---
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                mut_idx = np.random.choice(nonsens_idx)
                pert = np.random.uniform(-0.1 * col_range[mut_idx], 0.1 * col_range[mut_idx])
                new_pop[i, mut_idx] = np.clip(new_pop[i, mut_idx] + pert,
                                               col_min[mut_idx], col_max[mut_idx])
                if mut_idx in int_cols:
                    new_pop[i, mut_idx] = round(new_pop[i, mut_idx])

        population = new_pop

    # Evaluate final generation if budget remains
    if total_evals < ga_budget:
        for i in range(pop_size):
            if total_evals >= ga_budget:
                break
            sample_a = pd.Series(population[i], index=columns)
            sample_b = _flip_sensitive_single(sample_a, X_test, sensitive_cols)
            is_disc, fit = _evaluate_single(model, sample_a, sample_b, threshold)
            total_evals += 1
            disc_count += is_disc
            if is_disc:
                disc_samples_phase1.append(population[i].copy())

    # --- Phase 2: Local Search around discriminatory instances ---
    if len(disc_samples_phase1) == 0:
        remaining = num_samples - total_evals
        if remaining > 0:
            rs_disc, rs_n = random_search(model, X_test, sensitive_cols, non_sensitive_cols,
                                          num_samples=remaining, threshold=threshold)
            disc_count += rs_disc
            total_evals += rs_n
        return disc_count, total_evals

    seeds = np.array(disc_samples_phase1)
    local_remaining = num_samples - total_evals
    if local_remaining <= 0:
        return disc_count, total_evals

    # Local search: perturb non-sensitive features of seed samples, evaluate one by one
    seed_idx = 0
    for _ in range(local_remaining):
        seed = seeds[seed_idx % len(seeds)].copy()
        seed_idx += 1

        # Small perturbation on non-sensitive features (5%)
        for ci in nonsens_idx:
            pert = np.random.uniform(-0.05 * col_range[ci], 0.05 * col_range[ci])
            seed[ci] = np.clip(seed[ci] + pert, col_min[ci], col_max[ci])
            if ci in int_cols:
                seed[ci] = round(seed[ci])

        sample_a = pd.Series(seed, index=columns)
        sample_b = _flip_sensitive_single(sample_a, X_test, sensitive_cols)
        is_disc, _ = _evaluate_single(model, sample_a, sample_b, threshold)
        total_evals += 1
        disc_count += is_disc

    return disc_count, total_evals


# ============================================================
# Run experiment for one dataset
# ============================================================
def run_single_experiment(method_fn, method_name, model, X_test,
                          sensitive_cols, non_sensitive_cols,
                          num_samples=1000, num_runs=30, threshold=0.05,
                          desc=""):
    """Run a method num_runs times, return raw IDI ratios and times."""
    idi_ratios = []
    times = []
    for _ in tqdm(range(num_runs), desc=desc, leave=False):
        start = time.time()
        disc, total = method_fn(model, X_test, sensitive_cols, non_sensitive_cols,
                                num_samples=num_samples, threshold=threshold)
        elapsed = time.time() - start
        idi_ratios.append(disc / total if total > 0 else 0.0)
        times.append(elapsed)
    return idi_ratios, times


def process_dataset(dataset):
    """Process one dataset: run both Random Search and GA+LS, for Combined and Single scenarios."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print(f"[PID {os.getpid()}] Processing: {dataset['name']}")

    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset['file'], dataset['target'])
    model = load_model(dataset['model'])

    all_columns = list(X_test.columns)
    sensitive_cols = dataset['sensitive']
    non_sensitive_cols = [c for c in all_columns if c not in sensitive_cols]

    results = []
    methods = [
        ('Random Search', random_search),
        ('GA+Local Search', ga_local_search),
    ]

    # --- Combined scenario ---
    for method_name, method_fn in methods:
        print(f"  [{dataset['name']}] {method_name} - Combined")
        idi_ratios, times = run_single_experiment(
            method_fn, method_name, model, X_test, sensitive_cols, non_sensitive_cols,
            desc=f"{dataset['name']} {method_name} Combined"
        )
        results.append({
            'Dataset': dataset['name'],
            'Method': method_name,
            'Scenario': 'Combined',
            'Sensitive_Features_Used': ', '.join(sensitive_cols),
            'IDI_Ratios': idi_ratios,
            'Times': times,
            'Average_IDI_Ratio': np.mean(idi_ratios),
            'Average_Time_Seconds': np.mean(times),
            'Total_Runs': 30
        })

    # --- Single scenario (each sensitive feature) ---
    for sens_feat in sensitive_cols:
        for method_name, method_fn in methods:
            print(f"  [{dataset['name']}] {method_name} - Single({sens_feat})")
            idi_ratios, times = run_single_experiment(
                method_fn, method_name, model, X_test, [sens_feat], non_sensitive_cols,
                desc=f"{dataset['name']} {method_name} {sens_feat}"
            )
            results.append({
                'Dataset': dataset['name'],
                'Method': method_name,
                'Scenario': 'Single',
                'Sensitive_Features_Used': sens_feat,
                'IDI_Ratios': idi_ratios,
                'Times': times,
                'Average_IDI_Ratio': np.mean(idi_ratios),
                'Average_Time_Seconds': np.mean(times),
                'Total_Runs': 30
            })

    print(f"  [{dataset['name']}] Done!")
    return results


def main():
    num_workers = min(mp.cpu_count(), len(datasets))
    print(f"Running with {num_workers} workers across {len(datasets)} datasets")
    print(f"Methods: Random Search (baseline) vs GA+Local Search (improved)")
    print()

    with mp.Pool(processes=num_workers) as pool:
        all_results = pool.map(process_dataset, datasets)

    results = [r for sublist in all_results for r in sublist]

    # --- Save summary results ---
    out_file = 'improved_experiment_results.csv'
    with open(out_file, 'w', newline='') as f:
        fieldnames = ['Dataset', 'Method', 'Scenario', 'Sensitive_Features_Used',
                      'Average_IDI_Ratio', 'Average_Time_Seconds', 'Total_Runs']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: v for k, v in row.items() if k in fieldnames})
    print(f"\nSummary results saved to {out_file}")

    # --- Save raw per-run data ---
    raw_file = 'improved_experiment_raw.csv'
    with open(raw_file, 'w', newline='') as f:
        fieldnames = ['Dataset', 'Method', 'Scenario', 'Sensitive_Features_Used',
                      'Run', 'IDI_Ratio', 'Time_Seconds']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            for run_idx in range(len(row['IDI_Ratios'])):
                writer.writerow({
                    'Dataset': row['Dataset'],
                    'Method': row['Method'],
                    'Scenario': row['Scenario'],
                    'Sensitive_Features_Used': row['Sensitive_Features_Used'],
                    'Run': run_idx + 1,
                    'IDI_Ratio': row['IDI_Ratios'][run_idx],
                    'Time_Seconds': row['Times'][run_idx]
                })
    print(f"Raw per-run data saved to {raw_file}")

    # --- Wilcoxon Signed-Rank Test (Random Search vs GA+LS) ---
    print(f"\n{'='*85}")
    print(f"{'Dataset':<15} {'Scenario':<20} {'RS IDI':>8} {'GA IDI':>8} {'p-value':>10} {'Significant?':>13}")
    print(f"{'-'*85}")

    stat_rows = []
    for ds in datasets:
        scenarios = [('Combined', ', '.join(ds['sensitive']))]
        for sf in ds['sensitive']:
            scenarios.append(('Single', sf))

        for scenario, sens_used in scenarios:
            rs = [r for r in results if r['Dataset'] == ds['name']
                  and r['Scenario'] == scenario and r['Sensitive_Features_Used'] == sens_used
                  and r['Method'] == 'Random Search']
            ga = [r for r in results if r['Dataset'] == ds['name']
                  and r['Scenario'] == scenario and r['Sensitive_Features_Used'] == sens_used
                  and r['Method'] == 'GA+Local Search']

            if rs and ga:
                rs_idis = np.array(rs[0]['IDI_Ratios'])
                ga_idis = np.array(ga[0]['IDI_Ratios'])
                diff = ga_idis - rs_idis

                # Wilcoxon requires non-zero differences
                if np.all(diff == 0):
                    p_value = 1.0
                else:
                    _, p_value = wilcoxon(rs_idis, ga_idis)

                sig = 'Yes' if p_value < 0.05 else 'No'
                label = f"{scenario}({sens_used})" if scenario == 'Single' else scenario
                print(f"{ds['name']:<15} {label:<20} {np.mean(rs_idis):>8.4f} {np.mean(ga_idis):>8.4f} {p_value:>10.2e} {sig:>13}")

                stat_rows.append({
                    'Dataset': ds['name'],
                    'Scenario': scenario,
                    'Sensitive_Features_Used': sens_used,
                    'RS_Mean_IDI': np.mean(rs_idis),
                    'GA_Mean_IDI': np.mean(ga_idis),
                    'Wilcoxon_p_value': p_value,
                    'Significant_at_0.05': sig
                })

    print(f"{'='*85}")

    # Save statistical test results
    stat_file = 'statistical_test_results.csv'
    with open(stat_file, 'w', newline='') as f:
        fieldnames = ['Dataset', 'Scenario', 'Sensitive_Features_Used',
                      'RS_Mean_IDI', 'GA_Mean_IDI', 'Wilcoxon_p_value', 'Significant_at_0.05']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stat_rows:
            writer.writerow(row)
    print(f"Statistical test results saved to {stat_file}")


if __name__ == "__main__":
    main()
