# Explaining outliers - Isolation forest with shapley interactions (SHAP-IQ)

This repository contains the code for the experiments for the paper: Explaining Outliers using Isolation Forest and Shapley Interactions (Visser et al. 2025).
This paper has been accepted for publication, and will be presented, at ESANN 2025.


## Use method

Method has been added to the SHAP-IQ package which can be installed through pip:

```
pip install shapiq
```

Below you can see a basic usage example to generate explanations and visualize them. Check the [shapiq documentation](https://shapiq.readthedocs.io/en/latest/) for more details.
```
from sklearn.ensemble import IsolationForest
import shapiq

# Train isolation forest
clf = IsolationForest(n_estimators=200, contamination='auto', random_state=None, bootstrap=False)
clf.fit(X_train)

# Create shapiq treeexplainer for model
explainer = shapiq.TreeExplainer(model=clf, index="k-SII", min_order=1, max_order=3)

# Generate feature interaction explanations
x = X_test[0]
interaction_values = explainer.explain(x)

# Visualize explanations
shapiq.network_plot(
    first_order_values=interaction_values.get_n_order_values(1),
    second_order_values=interaction_values.get_n_order_values(2),
    feature_names=column_names # Where column names are a list of column names used for printing
)
```

## Experiments

In order to run the paper experiments follow the instructions below.

### Setup, explanation generation, and evaluation

- install requirements (full environments available in requirements_full.txt and requirements_full_conda.txt)
- run experiments for synthetic data (see below) and real-world dataset (experiment_realworld.py)
- run evaluation scripts for synthetic (synthetic_evaluation.py) and real-world (glass_evaluation.py) data experiments.

### Run & create synthetic experiment

- `python experiment.py diagonal_feature --data_type "diagonal_feature" --n_trials 10 --results_folder "./results"`
- `python experiment.py single_outlier_quandrant --data_type "single_outlier_quandrant" --n_trials 10 --results_folder "./results"`

## Paper & Citation

Preprint at: 

```bib
```
