# Explaining outliers - Isolation forest with shapley interactions (SHAP-IQ)

This repository contains the code for the experiments for the paper: Explaining Outliers using Isolation Forest and Shapley Interactions (Visser et al. 2025).
This paper has been accepted for publication, and will be presented, at ESANN 2025.

## Setup, explanation generation, and evaluation

- install requirements (full environments available in requirements_full.txt and requirements_full_conda.txt)
- run experiments for synthetic data (see below) and real-world dataset (experiment_realworld.py)
- run evaluation scripts for synthetic (synthetic_evaluation.py) and real-world (glass_evaluation.py) data experiments.

## Run & create synthetic experiment

- `python experiment.py diagonal_feature --data_type "diagonal_feature" --n_trials 10 --results_folder "./results"`
- `python experiment.py single_outlier_quandrant --data_type "single_outlier_quandrant" --n_trials 10 --results_folder "./results"`

## Paper & Citation

Preprint at: 

```bib
```