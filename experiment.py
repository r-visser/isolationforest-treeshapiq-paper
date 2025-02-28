import argparse
import json
import os
import pickle

import numpy as np
import ray
from sklearn.model_selection import ParameterGrid

from synthetic_data import (CorrelatedFeaturesGenerator,
                            SingleFeatureDiagonalGenerator,
                            SingleOutlierQuadrantGenerator, XORGenerator)
from trials import (IsoForestDIFFITrial, IsoForestShapIQTrial,
                    IsoForestShapTrial, TrialData)


def summarize_trial_results(trials_results:list):
    shap_means = np.array([trial["shap_mean"] for trial in trials_results])
    shap_stds = np.array([trial["shap_std"] for trial in trials_results])
    diffi_means = np.array([trial["diffi_mean"] for trial in trials_results])
    diffi_stds = np.array([trial["diffi_std"] for trial in trials_results])
    shap_iq_means = np.array([trial["shap_iq_mean"] for trial in trials_results])
    shap_iq_stds = np.array([trial["shap_iq_std"] for trial in trials_results])

    isoforest_performances = [trial["isoforest_performance"] for trial in trials_results]
    isoforest_accuracies = [performance["accuracy"] for performance in isoforest_performances]
    isoforest_precisions = [performance["precision"] for performance in isoforest_performances]
    isoforest_recalls = [performance["recall"] for performance in isoforest_performances]
    isoforest_f1s = [performance["f1"] for performance in isoforest_performances]

    return {
        "shap_mean": np.mean(shap_means),
        "shap_std": combined_standard_deviation(shap_means, shap_stds),
        "diffi_mean": np.mean(diffi_means),
        "diffi_std": combined_standard_deviation(diffi_means, diffi_stds),
        "shap_iq_mean": np.mean(shap_iq_means),
        "shap_iq_std": combined_standard_deviation(shap_iq_means, shap_iq_stds),
        "isoforest_performance": {
            "accuracy": np.mean(isoforest_accuracies),
            "accuracy_std": np.std(isoforest_accuracies),
            "precision": np.mean(isoforest_precisions),
            "precision_std": np.std(isoforest_precisions),
            "recall": np.mean(isoforest_recalls),
            "recall_std": np.std(isoforest_recalls),
            "f1": np.mean(isoforest_f1s),
            "f1_std": np.std(isoforest_f1s)
        }
    }

def combined_standard_deviation(means, stds):
    """
    Calculate the combined standard deviation from multiple groups.

    Parameters:
    means (list or np.array): Array of means for each group.
    stds (list or np.array): Array of standard deviations for each group.

    Returns:
    float: Combined standard deviation.
    """
    overall_mean = np.mean(means)
    term_within_groups = np.mean(stds**2)
    term_between_groups = np.mean((means - overall_mean)**2)
    return np.sqrt(term_within_groups + term_between_groups)

def run_trials_for_dataset(data: TrialData, save_path=None):
    print("Running trials for dataset")
    print("shap")
    trial_SHAP = IsoForestShapTrial(data)
    trial_SHAP.run_experiment(print_scores=False)
    isoforest = trial_SHAP.clf
    if_acc, if_prec, if_recall, if_f1 = trial_SHAP.isoforest_performance
    shap_mean, shap_std = trial_SHAP.evaluate_explanation_quality()

    print("diffi")
    trial_DIFFI = IsoForestDIFFITrial(data)
    trial_DIFFI.clf = isoforest
    trial_DIFFI.run_experiment(print_scores=False)
    diffi_mean, diffi_std = trial_DIFFI.evaluate_explanation_quality()

    print("shap-iq")
    trial_iq = IsoForestShapIQTrial(data)
    trial_iq.clf = isoforest
    trial_iq.run_experiment(print_scores=False)
    shap_iq_mean, shap_iq_std = trial_iq.evaluate_explanation_quality()

    result_scores = {
        "shap_mean": shap_mean,
        "shap_std": shap_std,
        "diffi_mean": diffi_mean,
        "diffi_std": diffi_std,
        "shap_iq_mean": shap_iq_mean,
        "shap_iq_std": shap_iq_std,
        "isoforest_performance": {
            "accuracy": if_acc,
            "precision": if_prec,
            "recall": if_recall,
            "f1": if_f1
        }
    }

    if save_path:
        # Create directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save data
        data.save_to_path(save_path)
        # Save model
        with open(os.path.join(save_path, "isoforest_model.pkl"), "wb") as f:
            pickle.dump(isoforest, f)
        # Save result scores to json file
        with open(os.path.join(save_path, f"trial_results.json"), "w") as f:
            json.dump(result_scores, f)
        # Save trial objects
        trial_SHAP.save_explanations(save_path + "/shap")
        trial_DIFFI.save_explanations(save_path + "/diffi")
        trial_iq.save_explanations(save_path + "/shap_iq")

    return result_scores

@ray.remote
def run_XOR_trial(trial_params, save_path=None):
    if save_path is not None:
        trial_id = trial_params.get("trial_id", 0)
        save_path += f"/trial_{trial_id}"

    XORgenerator = XORGenerator(noise_level=0.01)
    data = XORgenerator.generate_inliers_outliers(n_inliers=2500, n_outliers=100, n_random_features=2)
    trial_results = run_trials_for_dataset(data, save_path=save_path)

    return trial_results

def run_single_feature_XOR_trial(trial_params, save_path=None, direction="horizontal"):
    if save_path is not None:
        trial_id = trial_params.get("trial_id", 0)
        save_path += f"/trial_{trial_id}"

    XORgenerator = XORGenerator(noise_level=0.01)
    data = XORgenerator.generate_single_feature_inliers_outliers(n_inliers=2500, n_outliers=100, direction=direction)
    data.add_random_features(n_features=2)
    trial_results = run_trials_for_dataset(data, save_path=save_path)

    return trial_results

@ray.remote
def run_XOR_horizontal_trial(trial_params, save_path=None):
    return run_single_feature_XOR_trial(trial_params, save_path=save_path, direction="horizontal")

@ray.remote
def run_XOR_vertical_trial(trial_params, save_path=None):
    return run_single_feature_XOR_trial(trial_params, save_path=save_path, direction="vertical")

@ray.remote
def run_diagonal_feature_trial(trial_params, save_path=None):
    if save_path is not None:
        trial_id = trial_params.get("trial_id", 0)
        save_path += f"/trial_{trial_id}"

    SFGenerator = SingleFeatureDiagonalGenerator(noise_level=0.1)
    data = SFGenerator.generate_inliers_outliers(n_inliers=2500, n_outliers=50, n_random_features=10)
    # data.add_random_features(n_features=10)
    trial_results = run_trials_for_dataset(data, save_path=save_path)

    return trial_results

@ray.remote
def run_single_outlier_quandrant_trial(trial_params, save_path=None):
    if save_path is not None:
        trial_id = trial_params.get("trial_id", 0)
        save_path += f"/trial_{trial_id}"

    SOGenerator = SingleOutlierQuadrantGenerator(noise_level=0.1, outlier_location=[0, 1])
    data = SOGenerator.generate_inliers_outliers(n_inliers=2500, n_outliers=50, n_random_features=10)
    trial_results = run_trials_for_dataset(data, save_path=save_path)

    return trial_results


@ray.remote
def run_correlated_features_trial(trial_params):
    correlation_strength=trial_params.get("correlation_strength", 0.97)
    perturbation_magnitude=trial_params.get("perturbation_magnitude", 3.5)
    min_perturbation=trial_params.get("min_perturbation", 2.5)
    add_noise=trial_params.get("add_noise", True)
    noise_level=trial_params.get("noise_level", 0.1)
    CFGenerator = CorrelatedFeaturesGenerator(
        correlation_strength=correlation_strength, 
        perturbation_magnitude=perturbation_magnitude, 
        min_perturbation=min_perturbation, 
        add_noise=add_noise, 
        noise_level=noise_level
    )
    data = CFGenerator.generate_inliers_outliers(n_inliers=2500, n_outliers=100, n_random_features=10)
    
    print("shap-iq ", trial_params)
    trial_iq = IsoForestShapIQTrial(data)
    trial_iq.run_experiment(print_scores=False)
    shap_iq_mean, shap_iq_std = trial_iq.evaluate_explanation_quality()
    if_acc, if_prec, if_recall, if_f1 = trial_iq.isoforest_performance
    trial_results = {
        "params": trial_params,
        "shap_iq_mean": shap_iq_mean,
        "shap_iq_std": shap_iq_std,
        "isoforest_performance": {
            "accuracy": if_acc,
            "precision": if_prec,
            "recall": if_recall,
            "f1": if_f1
        }
    }
    # trial_results = run_trials_for_dataset(data)
    return trial_results

@ray.remote
def run_comparison_trial(trial_params):
    XORgenerator = XORGenerator(noise_level=0.1)
    CFGenerator = CorrelatedFeaturesGenerator(
        correlation_strength=0.97, 
        perturbation_magnitude=3.5, 
        min_perturbation=2.5, 
        add_noise=True, 
        noise_level=0.1
    )

    data = XORgenerator.generate_inliers_outliers(n_inliers=2500, n_outliers=100, n_random_features=0)
    data.add_outliers("XOR", XORgenerator, n_outliers=100)
    data.add_outliers("Cor", CFGenerator, n_outliers=100)
    data.add_random_features(n_features=10)

    trial_results = run_trials_for_dataset(data)
    return trial_results

def model_comparison_experiment(n_trials, n_outliers=None):
    ray.shutdown()
    ray.init()
    params = {}
    # run trials using different datasets
    trials = []
    for i in range(n_trials):
        print("Running trial ", i)
        trials.append(run_comparison_trial.remote(params))

    trial_ouputs = ray.get(trials)  # Evaluate remote functions (i.e. trials with different parameterizations)
    results = summarize_trial_results(trial_ouputs)

    return results

def correlated_features_experiment(n_trials):
    ray.shutdown()
    ray.init()
    param_grid = {
        "correlation_strength": [0.9, 0.93, 0.95, 0.97, 0.99], 
        # "perturbation_magnitude": [1.5, 2.5, 3.5, 4.5], 
        # "min_perturbation": [0.5, 1.0, 1.5, 2.0, 2.5], 
        # "add_noise": [True], 
        # "noise_level": [0.1]
    }

    trials = []
    for params in ParameterGrid(param_grid):
        print("Running trial ", params)
        trials.append(
            run_correlated_features_trial.remote(params))
            
    trial_ouputs = ray.get(trials)  # Evaluate remote functions (i.e. trials with different parameterizations)

    results = {}
    for trial_output in trial_ouputs:
        correlation_strength = trial_output["params"]["correlation_strength"]
        results[correlation_strength] = trial_output

    return results

def run_experiment(trial_function, results_folder, n_trials):
    ray.shutdown()
    ray.init()

    trials = []
    for i in range(n_trials):
        print("Running trial ", i)
        trials.append(trial_function.remote({"trial_id": i}, save_path=results_folder))

    trial_outputs = ray.get(trials)  # Evaluate remote functions (i.e. trials with different parameterizations)

    results = summarize_trial_results(trial_outputs)
    # Save results to file
    with open(f"{results_folder}/experiment_results.json", "w") as f:
        json.dump(results, f)

    ray.shutdown()
    
    return results

def run_XOR_experiment(results_folder, n_trials):
    return run_experiment(run_XOR_trial, results_folder, n_trials)

def run_XOR_horizontal_experiment(results_folder, n_trials):
    return run_experiment(run_XOR_horizontal_trial, results_folder, n_trials)

def run_XOR_vertical_experiment(results_folder, n_trials):
    return run_experiment(run_XOR_vertical_trial, results_folder, n_trials)

def run_diagonal_feature_experiment(results_folder, n_trials):
    return run_experiment(run_diagonal_feature_trial, results_folder, n_trials)

def run_single_outlier_quandrant_experiment(results_folder, n_trials):
    return run_experiment(run_single_outlier_quandrant_trial, results_folder, n_trials)

if __name__ == "__main__":
    # Read parameters from command line
    parser = argparse.ArgumentParser(description='Run model experiment.')
    parser.add_argument('--experiment_name', type=str, nargs='?', default="", help='Name of the experiment.')
    parser.add_argument('--data_type', type=str, default="XOR", help='Type of data to generate.')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials to run.')
    parser.add_argument('--results_folder', type=str, default="./results", help='Root folder to store results.')
    args = parser.parse_args()

    # if experiment name not provided use data type
    if args.experiment_name == "":
        args.experiment_name = args.data_type

    print(f"Experiment Name: {args.experiment_name}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Results Folder: {args.results_folder}")

    # Create the full path for the results folder
    full_results_folder = f"{args.results_folder}/{args.experiment_name}"

    # Check if folder exists if so create change name by adding a number
    if os.path.exists(full_results_folder):
        i = 1
        while os.path.exists(f"{full_results_folder}_{i}"):
            i += 1
        full_results_folder = f"{full_results_folder}_{i}"
    
    os.makedirs(full_results_folder)

    # Save experimental settings to file
    with open(f"{full_results_folder}/experiment_settings.json", "w") as f:
        json.dump(vars(args), f)

    # Run the experiment
    if args.data_type == "XOR":
        run_XOR_experiment(full_results_folder, args.n_trials)
    elif args.data_type == "XOR_horizontal":
        run_XOR_horizontal_experiment(full_results_folder, args.n_trials)
    elif args.data_type == "XOR_vertical":
        run_XOR_vertical_experiment(full_results_folder, args.n_trials)
    elif args.data_type == "diagonal_feature":
        run_diagonal_feature_experiment(full_results_folder, args.n_trials)
    elif args.data_type == "single_outlier_quandrant":
        run_single_outlier_quandrant_experiment(full_results_folder, args.n_trials)
    elif args.data_type == "correlated_features":
        correlated_features_experiment(args.n_trials)
    elif args.data_type == "model_comparison":
        model_comparison_experiment(args.n_trials)
    else:
        raise ValueError("Invalid experiment name.")