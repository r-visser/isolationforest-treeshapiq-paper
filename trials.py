from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

import shapiq.shapiq as shapiq
from DIFFI.utils import local_diffi_batch
from helpers import (calculate_outlier_scores, create_interaction_labels,
                     print_outlier_scores)

if TYPE_CHECKING:
    from synthetic_data import DataGenerator

@dataclass
class OutlierSet():
    name: str
    idx: list
    outlier_feature_labels: list
    train_idx: list = None
    test_idx: list = None
    train_test_split_fixed: bool = False

    def to_json(self):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(asdict(self), default=convert, indent=4)
    
    def save_to_file(self, file_path):
        with open(file_path, 'w') as file:
            file.write(self.to_json())

    def load_from_file(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return OutlierSet(**data)

@dataclass
class TrialData():
    X: np.ndarray
    y: np.ndarray
    feature_labels: list
    outlier_sets: list[OutlierSet]
    ground_truth_outlier_features: np.ndarray = None
    X_train: np.ndarray = None
    y_train: np.ndarray = None
    X_test: np.ndarray = None
    y_test: np.ndarray = None
    ground_truth_outlier_features_test: np.ndarray = None
    outlier_idx: np.ndarray = None
    inlier_idx: np.ndarray = None

    def __post_init__(self):
        self.outlier_idx = np.where(self.y == -1)[0]
        self.inlier_idx = np.where(self.y == 1)[0]
        self.generate_ground_truth_outlier_features()

        if self.X_train is None:
            self.create_train_test_split()

    def save_to_path(self, folder_path, file_name=None):
        if file_name is None:
            file_name = "trial_data"
        np.savez(f"{folder_path}/{file_name}.npz",
                 X=self.X,
                 y=self.y,
                 feature_labels=self.feature_labels,
                 ground_truth_outlier_features=self.ground_truth_outlier_features,
                 X_train=self.X_train,
                 y_train=self.y_train,
                 X_test=self.X_test,
                 y_test=self.y_test,
                 ground_truth_outlier_features_test=self.ground_truth_outlier_features_test,
                 outlier_idx=self.outlier_idx,
                 inlier_idx=self.inlier_idx)
        
        for i, outlier_set in enumerate(self.outlier_sets):
            with open(f"{folder_path}/{file_name}_outlier_set_{i}.json", 'w') as file:
                file.write(outlier_set.to_json())

    @staticmethod
    def load_from_path(folder_path, file_name=None):
        if file_name is None:
            file_name = "trial_data"
        data = np.load(f"{folder_path}/{file_name}.npz")
        X = data['X']
        y = data['y']
        feature_labels = data['feature_labels'].tolist()
        ground_truth_outlier_features = data['ground_truth_outlier_features']
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        ground_truth_outlier_features_test = data['ground_truth_outlier_features_test']
        outlier_idx = data['outlier_idx']
        inlier_idx = data['inlier_idx']

        outlier_sets = []
        
        for file in os.listdir(folder_path):
            if file.startswith(file_name) and file.endswith(".json"):
                outlier_set = OutlierSet.load_from_file(f"{folder_path}/{file}")
                outlier_sets.append(outlier_set)
                
        return TrialData(X=X, y=y, feature_labels=feature_labels, outlier_sets=outlier_sets,
                         ground_truth_outlier_features=ground_truth_outlier_features,
                         X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                         ground_truth_outlier_features_test=ground_truth_outlier_features_test,
                         outlier_idx=outlier_idx, inlier_idx=inlier_idx)

    @property
    def n_features(self):
        return self.X.shape[1]
    
    @property
    def inliers(self):
        return self.X[self.inlier_idx]
    
    @property
    def outliers(self):
        return self.X[self.outlier_idx]
    
    def generate_ground_truth_outlier_features(self):
        # create datafame for X filled with zeros with columns for each feature_label
        df = pd.DataFrame(np.zeros((len(self.X), len(self.feature_labels)), dtype=int), columns=self.feature_labels)
        # set values for outliers to 1 loop over outlier_sets
        for outlier_set in self.outlier_sets:
            # print(self.feature_labels)
            # print(outlier_set.outlier_feature_labels)
            mask = [column_name in outlier_set.outlier_feature_labels for column_name in self.feature_labels]
            df.loc[outlier_set.idx, mask] = 1
        # return the dataframe
        self.ground_truth_outlier_features = df.values
        self.ground_truth_outlier_features_df = df
        return df
        

    def create_train_test_split(self, test_size=0.8, train_size=None, random_state=0):
        if isinstance(test_size, int) and test_size >= len(self.outlier_idx)-1:
            raise ValueError("test_size should be less than the number of outliers")    

        outlier_test_idx = []
        outlier_train_idx = []
        for outlier_set in self.outlier_sets:
            if outlier_set.train_test_split_fixed:
                outlier_test_idx.extend(outlier_set.test_idx)
                outlier_train_idx.extend(outlier_set.train_idx)
            else:
                if isinstance(test_size, int):
                    size = test_size
                else:
                    size = int(test_size * len(outlier_set.idx))
                outlier_set.test_idx = np.random.choice(outlier_set.idx, size=size, replace=False)
                outlier_set.train_idx = np.setdiff1d(outlier_set.idx, outlier_set.test_idx)
                outlier_test_idx.extend(outlier_set.test_idx)
                outlier_train_idx.extend(outlier_set.train_idx)

        self.outlier_test_idx = np.array(outlier_test_idx)
        self.outlier_train_idx = np.array(outlier_train_idx)

        # create training and test sets
        self.X_train = np.concatenate([self.X[self.inlier_idx], self.X[self.outlier_train_idx]])
        self.y_train = np.concatenate([np.ones(len(self.inlier_idx)), -np.ones(len(self.outlier_train_idx))])
        self.X_test = self.X[self.outlier_test_idx]
        self.y_test = np.ones(len(self.outlier_test_idx)) * -1
        self.ground_truth_outlier_features_test = self.ground_truth_outlier_features[self.outlier_test_idx]

    def ground_truth_outlier_features_first_order_only(self, test_only=False):
        if test_only:
            return self.ground_truth_outlier_features_test[:, :self.n_features]
        else:
            return self.ground_truth_outlier_features[:, :self.n_features]
        
    def ground_truth_outlier_features_interaction_only(self, test_only=False):
        if test_only:
            return self.ground_truth_outlier_features_test[:, self.n_features:]
        else:
            return self.ground_truth_outlier_features[:, self.n_features:]
    
    def split_feature_labels(self):
        """Return all single features and all interactions."""
        # Find the index of the first tuple
        split_index = next(i for i, item in enumerate(self.feature_labels) if item.startswith('('))

        # Split the array
        features = self.feature_labels[:split_index]
        interactions = self.feature_labels[split_index:]
        return features, interactions
        
    def add_outliers(self, outlier_set_name, generator: DataGenerator, n_outliers=100):
        # Add outlier to existing data
        # 1. Generate inlier features for data in X
        # 2. Add inlier features to outlier set for existing data features in X
        if outlier_set_name is None:
            outlier_set_name = type(generator).__name__

        n_features = self.n_features
        
        # check if outlier set with the same name already exists
        # or with postfix _A, _B, ...
        i = 0
        base_name = outlier_set_name
        while any(outlier_set.name == outlier_set_name for outlier_set in self.outlier_sets):
            i += 1
            outlier_set_name = f"{base_name}_{chr(65+i)}"

        inlier_features, _ = generator.generate_inliers(len(self.X))
        outlier_features, _ = generator.generate_outliers(n_outliers)
        n_new_features = outlier_features.shape[1]

        # append inlier features to existing data
        self.X = np.concatenate([self.X, inlier_features], axis=1)
        # for subset of n_outliers inliers replace inlier features with outliers features
        new_outlier_idx = np.random.choice(self.inlier_idx, size=n_outliers, replace=False)
        self.X[new_outlier_idx, -n_new_features:] = outlier_features # replace inlier features with outlier features
        self.y[new_outlier_idx] = -1 # set labels to -1 for outliers
        self.inlier_idx = np.setdiff1d(self.inlier_idx, new_outlier_idx)
        self.outlier_idx = np.concatenate([self.outlier_idx, new_outlier_idx])
        
        # add feature labels and interaction labels to feature_labels
        features, interactions = self.split_feature_labels()
        # add new feature labels
        new_feature_names = [f"{outlier_set_name}_{i}" for i in range(n_new_features)]
        features.extend(new_feature_names)
        # add new interaction labels
        interactions = create_interaction_labels(features) # TODO fix only set single feature for SingleFeatureXORGenerator
        self.feature_labels = features + interactions

        self.outlier_sets.append(OutlierSet(name=outlier_set_name, idx=new_outlier_idx, outlier_feature_labels=new_feature_names)) # TODO fix outlier_feature_labels for SingleFeatureXORGenerator
        self.generate_ground_truth_outlier_features()

        self.create_train_test_split()        
    
    # def add_train_test_outliers(self, outlier_set_name, generator, n_outliers=100):
    #     # Add outlier to existing data
    #     # 1. Generate inlier features for data in X
    #     # 2. Add inlier features to outlier set for existing data features in X
    #     # 3. Add outliers to the training and test sets
    #     pass

    def add_random_features(self, n_features=10):
        # add random noise features
        self.X = np.column_stack([self.X, np.random.randn(self.X.shape[0], n_features)])
        feature_labels, _ = self.split_feature_labels()
        feature_labels += [f'Rnd_{i}' for i in range(n_features)]
        feature_labels += create_interaction_labels(feature_labels)
        self.feature_labels = feature_labels
        self.generate_ground_truth_outlier_features()
        self.create_train_test_split()


def compute_score(outlier_features_label, values, method:Literal["roc_auc", "cosine_similarity"]="roc_auc"):
    if method == "roc_auc":
        return roc_auc_score(y_true=outlier_features_label, y_score=values)
    elif method == "cosine_similarity":
        return cosine_similarity(outlier_features_label.reshape(1, -1), values.reshape(1, -1))[0, 0]
    return score

class IsoForestExplanationTrial(ABC):
    def __init__(self, data: TrialData, random_state=0):
        self.ds = data
        self.random_state = random_state
        self.clf = None
        self.explanations = {} # map outlier index to explanation

    def save_to_path(self, folder_path, file_name=None):
        if file_name is None:
            file_name = "trial"
        self.ds.save_to_path(folder_path, file_name)
        if self.clf is not None:
            np.save(f"{folder_path}/{file_name}_isoforest.npy", self.clf)

    def train_isolation_forest(self):
        # clf = IsolationForest(max_samples='auto', random_state=self.random_state)
        clf = IsolationForest(n_estimators=100, max_samples=64, contamination='auto', random_state=self.random_state, bootstrap=False)
        # clf = IsolationForest(n_estimators=200, contamination='auto', random_state=self.random_state, bootstrap=False)
        clf.fit(self.ds.X_train)
        self.clf = clf

    @abstractmethod
    def create_explainer(self):
        pass

    @abstractmethod
    def generate_explanation(self, idx):
        pass

    def get_explanation(self, idx):
        # If doesnt yet exist generate it otherwise look up
        if idx not in self.explanations:
            self.explanations[idx] = self.generate_explanation(idx)
        return self.explanations[idx]
    
    def save_explanations(self, folder_path):
        for idx in self.explanations:
            self.save_explanation(idx, folder_path)

    @abstractmethod
    def save_explanation(self, idx, folder_path):
        pass

    def run_experiment(self, print_scores=True):
        if self.clf is None:
            self.train_isolation_forest()
        # Predict outliers in the test set and compare with the true labels
        self.y_pred = self.clf.predict(self.ds.X_test)

        # F1 score
        if print_scores:
            self.isoforest_performance = print_outlier_scores(self.y_pred, self.ds.y_test)
        else:
            self.isoforest_performance = calculate_outlier_scores(self.y_pred, self.ds.y_test)
        

        self.create_explainer()

    @abstractmethod
    def evaluate_explanation_quality(self, only_predicted_outliers=True):
        pass

    def inspect_with_dropdown(self, true_outliers_only=False, type: Literal["network_plot", "bar_first_order", "bar_second_order", "force_plot"] = "network_plot"):
        # 7. Check if the explanations that are generated can retreive the perturbed features of the outliers
        # Select a predicted outlier to explain
        if true_outliers_only:
            outlier_indices = np.where((self.y_pred == -1) & (self.ds.y_test == -1))[0]
        else:
            outlier_indices = np.where(self.y_pred == -1)[0]
        print("Outlier Indices:\n", outlier_indices)

        # Create a dropdown menu for selecting an outlier index
        dropdown = widgets.Dropdown(
            options=outlier_indices,
            description='Outlier Index:',
            disabled=False,
        )

        def explain_outlier(change):
            outlier_index = change['new']
            interaction_values = self.get_explanation(outlier_index)
            
            # Clear previous output and display the new plot
            clear_output(wait=True)
            display(dropdown)
            
            # Print the actual label of the sample
            actual_label = self.ds.y_test[outlier_index]
            print(f"Actual label for the selected sample (index {outlier_index}): {actual_label}")
            
            column_names = self.ds.feature_labels[:self.ds.n_features]
            if type == "network_plot":
                shapiq.network_plot(
                    first_order_values=interaction_values.get_n_order_values(1),
                    second_order_values=interaction_values.get_n_order_values(2),
                    # TODO top k interaction
                    feature_names=self.ds.feature_labels,
                )
            elif type == "bar_first_order":
                shapiq.stacked_bar_plot(
                    interaction_values=interaction_values.get_n_order(1),
                    feature_names=column_names,
                )
            elif type == "bar_second_order" or type == "bar_plot":
                _ = shapiq.stacked_bar_plot(
                    interaction_values=interaction_values.get_n_order(2, min_order=1),
                    feature_names=column_names,
                )
            elif type == "force_plot":
                interaction_values.plot_force(
                    feature_names=column_names, feature_values=column_names, contribution_threshold=0.03
                )
            print(interaction_values.get_top_k_interactions(5))

        # Attach the function to the dropdown menu
        dropdown.observe(explain_outlier, names='value')

        # Display the dropdown menu
        display(dropdown)

        # Trigger the explanation for the initial value
        explain_outlier({'new': dropdown.value})


class IsoForestShapTrial(IsoForestExplanationTrial):
    def create_explainer(self):
        # self.shap_isoforest_clf = convert_values_isolation_forest_shap_isotree(self.clf)
        self.shap_isoforest_clf = self.clf
        self.explainer = shapiq.TreeExplainer(model=self.shap_isoforest_clf, index="SV", max_order=1)  # TODO SV and order 1 equivalent to SHAP values

    def generate_explanation(self, idx):
        x = self.ds.X_test[idx]
        interaction_values = self.explainer.explain(x)
        return interaction_values
    
    def save_explanation(self, idx, folder_path):
        self.explanations[idx].save(f"{folder_path}/explanation_{idx}.pkl")

    def evaluate_explanation_quality(self, only_predicted_outliers=True, method:Literal["roc_auc", "cosine_similarity"]="roc_auc"):
        """Computes the average ROC AUC score for the outlier features based on the SHAP interaction values."""
        if only_predicted_outliers:
            outlier_indices = np.where((self.y_pred == -1) & (self.ds.y_test == -1))[0]
        else:
            outlier_indices = np.where(self.ds.y_test == -1)[0]

        ground_truth_outlier_features_first_order_only = self.ds.ground_truth_outlier_features_first_order_only(test_only=True)

        scores = []
        for idx in outlier_indices:
            interaction_values = self.get_explanation(idx)
            shapley_values = np.absolute(interaction_values.get_n_order(1))
            outlier_features_label = ground_truth_outlier_features_first_order_only[idx]
            scores.append(compute_score(outlier_features_label, shapley_values, method))

        return np.mean(scores), np.std(scores)
    
    def inspect_with_dropdown(self, true_outliers_only=False):
        # 7. Check if the explanations that are generated can retreive the perturbed features of the outliers
        # Select a predicted outlier to explain
        if true_outliers_only:
            outlier_indices = np.where((self.y_pred == -1) & (self.ds.y_test == -1))[0]
        else:
            outlier_indices = np.where(self.y_pred == -1)[0]
        print("Outlier Indices:\n", outlier_indices)

        # Create a dropdown menu for selecting an outlier index
        dropdown = widgets.Dropdown(
            options=outlier_indices,
            description='Outlier Index:',
            disabled=False,
        )

        def explain_outlier(change):
            outlier_index = change['new']
            x = self.ds.X_test[outlier_index]
            interaction_values = self.explainer.explain(x)
            
            # Clear previous output and display the new plot
            clear_output(wait=True)
            display(dropdown)
            
            # Print the actual label of the sample
            actual_label = self.ds.y_test[outlier_index]
            print(f"Actual label for the selected sample (index {outlier_index}): {actual_label}")
            
            column_names = self.ds.feature_labels[:self.ds.n_features]
            # single features (1-order)
            _ = shapiq.stacked_bar_plot(
                interaction_values=interaction_values.get_n_order(1),
                feature_names=column_names,
            )
            print(interaction_values.get_top_k_interactions(5))

        # Attach the function to the dropdown menu
        dropdown.observe(explain_outlier, names='value')
        # Display the dropdown menu
        display(dropdown)
        # Trigger the explanation for the initial value
        explain_outlier({'new': dropdown.value})


class IsoForestShapIQTrial(IsoForestExplanationTrial):
    def create_explainer(self):
        # self.shap_isoforest_clf = convert_values_isolation_forest_shap_isotree(self.clf)
        self.shap_isoforest_clf = self.clf
        self.explainer = shapiq.TreeExplainer(model=self.shap_isoforest_clf, index="k-SII", min_order=1, max_order=2)

    def generate_explanation(self, idx):
        x = self.ds.X_test[idx]
        interaction_values = self.explainer.explain(x)
        return interaction_values
    
    def save_explanation(self, idx, folder_path):
        self.explanations[idx].save(f"{folder_path}/explanation_{idx}.pkl")

    def evaluate_explanation_quality(self, only_predicted_outliers=True, method:Literal["roc_auc", "cosine_similarity"]="roc_auc", ground_truth:Literal["all", "first_order_only", "interaction_only"]="all"):
        """Computes the average ROC AUC score for the outlier features based on the SHAP interaction values."""
        if only_predicted_outliers:
            outlier_indices = np.where((self.y_pred == -1) & (self.ds.y_test == -1))[0]
        else:
            outlier_indices = np.where(self.ds.y_test == -1)[0]

        ground_truth_outlier_features_test = self.ds.ground_truth_outlier_features_test.copy()
        if ground_truth == "first_order_only":
            # mask ground_truth_outlier_features_test interaction columns (set to 0)
            ground_truth_outlier_features_test[:, self.ds.n_features:] = 0
        elif ground_truth == "interaction_only":
            # mask ground_truth_outlier_features_test first order columns (set to 0)
            ground_truth_outlier_features_test[:, :self.ds.n_features] = 0

        scores = []
        for idx in outlier_indices:
            interaction_values = self.get_explanation(idx)
            scores_first_order = np.absolute(interaction_values.get_n_order(1))
            scores_second_order = np.absolute(interaction_values.get_n_order(2))
            shapley_values = np.concatenate([scores_first_order, scores_second_order])
            outlier_features_label = ground_truth_outlier_features_test[idx]
            scores.append(compute_score(outlier_features_label, shapley_values, method))

        return np.mean(scores), np.std(scores)


class IsoForestDIFFITrial(IsoForestExplanationTrial):
    def create_explainer(self):
        iforest = self.clf
        X_test = self.ds.X_test
        # y_te_pred = np.array(iforest.decision_function(X_test) < 0).astype('int')
        # self.diffi_test, self.ord_idx_diffi_test, exec_time_diffi_test = local_diffi_batch(iforest, X_test)
        # outlier_indices = np.where(self.y_pred == -1)[0]
        self.diffi_test, self.ord_idx_diffi_test, exec_time_diffi_test = local_diffi_batch(iforest, X_test[np.where(self.y_pred == -1)])
        # self.diffi_test, self.ord_idx_diffi_test, exec_time_diffi_test = local_diffi_batch(iforest, X_test)

        # plot_ranking_syn(ord_idx_diffi_te, title = 'Feature ranking glass dataset class 7 outliers - Local DIFFI')
        # print('Average computational time Local-DIFFI: {}'.format(round(np.mean(exec_time_diffi_test),3)))
        # return self.diffi_test, self.ord_idx_diffi_test, exec_time_diffi_test, self.y_pred
        for idx in range(len(self.diffi_test)):
            self.diffi_test[idx] = self.diffi_test[idx].flatten()

    def generate_explanation(self, idx):
        return self.diffi_test[idx]

    def get_explanation(self, idx):
        if idx not in self.explanations:
            self.explanations[idx] = self.diffi_test[idx]
        return self.explanations[idx]
    
    def save_explanation(self, idx, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # save ndarray to file
        np.save(f"{folder_path}/explanation_{idx}.npy", self.explanations[idx])

    def evaluate_explanation_quality(self, only_predicted_outliers=True, method:Literal["roc_auc", "cosine_similarity"]="roc_auc"):
        """Computes the average ROC AUC score for the outlier features based on the SHAP interaction values."""
        ground_truth = self.ds.ground_truth_outlier_features_first_order_only(test_only=True)[np.where(self.y_pred == -1)]
        scores = []
        for idx in range(len(self.diffi_test)):
            outlier_features_label = ground_truth[idx]
            diffi_explanation = self.get_explanation(idx)
            scores.append(compute_score(outlier_features_label, self.diffi_test[idx], method))

        return np.mean(scores), np.std(scores)