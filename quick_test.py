"""Quick test: Random Search vs GA+Local Search on 2 datasets, 3 runs each."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import time
from improved_experiment import (
    load_and_preprocess_data, random_search, ga_local_search
)
from tensorflow.keras.models import load_model

test_datasets = [
    {
        'name': 'ADULT',
        'file': 'ise-lab-solution/lab4/dataset/processed_adult.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_adult.h5',
        'target': 'Class-label',
        'sensitive': ['gender', 'race', 'age']
    },
    {
        'name': 'GERMAN',
        'file': 'ise-lab-solution/lab4/dataset/processed_german.csv',
        'model': 'ise-lab-solution/lab4/DNN/model_processed_greman_cleaned.h5',
        'target': 'CREDITRATING',
        'sensitive': ['PersonStatusSex', 'AgeInYears']
    },
]

NUM_RUNS = 3
NUM_SAMPLES = 1000

for ds in test_datasets:
    print(f"\n{'='*50}")
    print(f"Dataset: {ds['name']}")
    print(f"{'='*50}")

    X_train, X_test, _, _ = load_and_preprocess_data(ds['file'], ds['target'])
    model = load_model(ds['model'])
    sens = ds['sensitive']
    nonsens = [c for c in X_test.columns if c not in sens]

    for method_name, method_fn in [('Random Search', random_search), ('GA+Local Search', ga_local_search)]:
        idis, elapsed = [], []
        for _ in range(NUM_RUNS):
            t0 = time.time()
            disc, total = method_fn(model, X_test, sens, nonsens, num_samples=NUM_SAMPLES)
            elapsed.append(time.time() - t0)
            idis.append(disc / total)
        print(f"  {method_name:<20s}  IDI={np.mean(idis):.4f}  Time={np.mean(elapsed):.2f}s  (runs={NUM_RUNS})")
