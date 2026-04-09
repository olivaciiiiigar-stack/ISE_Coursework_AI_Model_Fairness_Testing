# ISE Coursework: AI Model Fairness Testing

This repository contains the code and data for the Intelligent Software Engineering coursework (Lab 4: AI Model Fairness Testing).

## Author

- **Name**: Yichi Zhang

## Problem

This project addresses **individual fairness testing** of pre-trained DNN models. The goal is to find as many individual discriminatory instances (IDI) as possible within a fixed budget of test inputs. The baseline is Random Search; we propose a **Genetic Algorithm + Local Search (GA+LS)** two-phase approach to improve the IDI ratio.

## Repository Structure

```
├── improved_experiment.py          # Main experiment script (GA+LS vs Random Search)
├── improved_experiment_results.csv # Summary results (average IDI ratio & time, 30 runs)
├── improved_experiment_raw.csv     # Raw per-run data (30 runs × all scenarios)
├── statistical_test_results.csv    # Wilcoxon signed-rank test results
├── requirements.txt               # Dependency description
├── pyproject.toml                 # uv project config with pinned dependencies
├── requirements.pdf               # Dependencies and versions
├── manual.pdf                     # How to use the tool
├── replication.pdf                # How to replicate the reported results
├── ise-lab-solution/lab4/
│   ├── dataset/                   # 8 pre-processed datasets (.csv)
│   ├── DNN/                       # 8 pre-trained DNN models (.h5)
│   └── lab4_solution.py           # Original baseline solution
└── ISE/lab4/                      # Lab description and raw datasets
```

## Datasets

| Dataset     | Domain      | Sensitive Features            | # Features |
|-------------|-------------|-------------------------------|------------|
| ADULT       | Finance     | gender, race, age             | 11         |
| COMPAS      | Criminology | Sex, Race                     | 13         |
| LAW SCHOOL  | Education   | male, race                    | 12         |
| KDD         | Criminology | sex, race                     | 19         |
| DUTCH       | Finance     | sex, age                      | 12         |
| CREDIT      | Finance     | SEX, EDUCATION, MARRIAGE      | 24         |
| CRIME       | Criminology | Black, FemalePctDiv           | 22         |
| GERMAN      | Finance     | PersonStatusSex, AgeInYears   | 20         |

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run the full experiment (30 runs × 8 datasets, ~2-3 hours)
uv run ./improved_experiment.py
```

## Output Files

After running `improved_experiment.py`, three CSV files are generated:

1. **`improved_experiment_results.csv`** — Summary table with average IDI ratio and time per method/dataset/scenario.
2. **`improved_experiment_raw.csv`** — Raw data: one row per run (Run 1–30) for each method/dataset/scenario.
3. **`statistical_test_results.csv`** — Wilcoxon signed-rank test p-values comparing Random Search vs GA+LS.

## Documentation

- **`requirements.pdf`**: Environment and dependency details.
- **`manual.pdf`**: Tool usage instructions.
- **`replication.pdf`**: Step-by-step instructions to replicate all reported results.
