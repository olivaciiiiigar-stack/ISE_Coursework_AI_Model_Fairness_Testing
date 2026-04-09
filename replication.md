# Replication Guide

This document provides step-by-step instructions to replicate all results reported in the coursework report.

## Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/olivaciiiiigar-stack/ISE_Coursework_AI_Model_Fairness_Testing.git
   cd ISE_Coursework_AI_Model_Fairness_Testing
   ```

2. **Install dependencies** (choose one):
   ```bash
   # Using uv (recommended)
   uv sync

   # Using pip
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   uv run python -c "import tensorflow; print(tensorflow.__version__)"
   # Expected output: 2.21.0
   ```

## Replicate the Full Experiment

Run the main experiment script:

```bash
uv run ./improved_experiment.py
```

### What this does

- Loads 8 pre-trained DNN models and their corresponding datasets from `ise-lab-solution/lab4/`.
- For each dataset, runs both **Random Search** (baseline) and **GA+Local Search** (proposed) under:
  - **Combined scenario**: all sensitive features tested together.
  - **Single scenario**: each sensitive feature tested individually.
- Each method/scenario combination is repeated **30 times**.
- Datasets are processed in parallel using Python multiprocessing (8 workers).

### Expected runtime

- **Full experiment**: approximately 2–3 hours (varies by CPU).
- **Quick test** (`uv run ./quick_test.py`): approximately 10 minutes (2 datasets, 3 runs).

### Output files

After completion, the following files are generated in the project root:

| File | Content |
|------|---------|
| `improved_experiment_results.csv` | Average IDI ratio and time per method/dataset/scenario |
| `improved_experiment_raw.csv` | Raw IDI ratio and time for each of the 30 runs |
| `statistical_test_results.csv` | Wilcoxon signed-rank test p-values |

### Console output

The script also prints a Wilcoxon test summary table to the console, showing per-dataset p-values and significance.

## Verifying Results

1. **Summary CSV**: Open `improved_experiment_results.csv`. Each row shows one method–dataset–scenario combination with the average IDI ratio over 30 runs and average execution time.

2. **Statistical tests**: Open `statistical_test_results.csv`. For each dataset/scenario, the `Wilcoxon_p_value` column shows the p-value from the Wilcoxon signed-rank test comparing Random Search vs GA+LS. `Significant_at_0.05 = Yes` indicates the difference is statistically significant.

3. **Raw data**: Open `improved_experiment_raw.csv` for per-run IDI ratios (30 rows per method/dataset/scenario), which can be used to reproduce box plots or further statistical analysis.

## Notes on Reproducibility

- The experiment uses **random sampling** without a fixed global seed, so exact IDI ratio values will vary slightly between runs. However, the overall trends (GA+LS outperforming Random Search on most datasets) and statistical significance should be consistent.
- The pre-trained models (`ise-lab-solution/lab4/DNN/*.h5`) and datasets (`ise-lab-solution/lab4/dataset/*.csv`) are included in the repository and are not re-trained during the experiment.
- Both methods use exactly the same single-sample `model.predict()` approach and the same budget (1000 samples per run) to ensure a fair comparison.
