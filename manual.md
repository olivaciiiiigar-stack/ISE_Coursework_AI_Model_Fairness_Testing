# Manual: AI Model Fairness Testing Tool

## Overview

This tool compares two fairness testing methods — **Random Search** (baseline) and **GA+Local Search** (proposed improvement) — on 8 pre-trained DNN binary classification models. It measures the Individual Discriminatory Instance (IDI) ratio: the proportion of generated test inputs that expose discriminatory behaviour in the model.

## Prerequisites

- Python 3.12+
- All dependencies listed in `requirements.txt`
- The pre-trained models and datasets are already included in the repository under `ise-lab-solution/lab4/`.

## Installation

```bash
# Clone the repository
git clone https://github.com/olivaciiiiigar-stack/ISE_Coursework_AI_Model_Fairness_Testing.git
cd ISE_Coursework_AI_Model_Fairness_Testing

# Install dependencies (Option A: using uv)
uv sync

# Install dependencies (Option B: using pip)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Full Experiment

```bash
uv run ./improved_experiment.py
```

This runs both methods (Random Search and GA+Local Search) on all 8 datasets, under two scenarios:

- **Combined**: All sensitive features are tested together.
- **Single**: Each sensitive feature is tested individually.

Each scenario is repeated **30 times** for statistical robustness. The experiment uses **multiprocessing** (one process per dataset) to run datasets in parallel.

**Expected runtime**: approximately 2–3 hours on a standard laptop (depends on CPU cores).

## Running a Quick Test

```bash
uv run ./quick_test.py
```

This runs only 2 datasets (ADULT, GERMAN) with 3 runs each, completing in ~10 minutes. Useful for verifying the setup.

## Output Files

After the full experiment completes, three CSV files are generated in the project root:

| File | Description |
|------|-------------|
| `improved_experiment_results.csv` | Summary table: average IDI ratio and average time per method, dataset, and scenario (one row per combination). |
| `improved_experiment_raw.csv` | Raw per-run data: one row per individual run (Run 1–30), for all method/dataset/scenario combinations. Suitable for plotting and further statistical analysis. |
| `statistical_test_results.csv` | Wilcoxon signed-rank test results comparing Random Search vs GA+LS for each dataset/scenario, including p-values and significance at α = 0.05. |

## Understanding the Results

- **IDI Ratio** (Individual Discriminatory Instance ratio): `number_of_discriminatory_instances / total_generated_inputs`. Higher = more fairness bugs found.
- **Method = Random Search**: Baseline that randomly generates sample pairs and tests them.
- **Method = GA+Local Search**: Two-phase approach — Phase 1 uses a Genetic Algorithm for global exploration (60% budget), Phase 2 uses Local Search to refine around discovered discriminatory regions (40% budget).
- **Scenario = Combined**: All sensitive features perturbed simultaneously.
- **Scenario = Single**: Only one sensitive feature perturbed at a time.

## Configuration

Key parameters in `improved_experiment.py` can be adjusted near the top of each method function:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | 1000 | Total budget (number of test input pairs generated per run) |
| `num_runs` | 30 | Number of independent repetitions per scenario |
| `threshold` | 0.05 | Minimum prediction difference to count as discriminatory |
| `pop_size` | 200 | GA population size |
| `ga_generations` | 5 | Number of GA generations in Phase 1 |
| `ga_ratio` | 0.6 | Fraction of budget allocated to GA phase (rest goes to Local Search) |
| `perturbation_rate` | 0.05 | Local Search perturbation step size |
