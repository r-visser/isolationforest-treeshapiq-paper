import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from DIFFI.utils import local_diffi_batch
from helpers import print_outlier_scores
from realworld_data import load_glass_data
from shapiq import shapiq
from shapiq.shapiq import InteractionValues

# Script to load all explanations
def load_glass_explanations():
    explanations = dict()
    folder = "results/Glass_explanations"
    for outlier_class in ["out_5", "out_6", "out_7"]:
        explanations[outlier_class] = dict()
        shap_folder = os.path.join(folder, outlier_class, "shap")
        shap_iq_folder = os.path.join(folder, outlier_class, "shap_iq")
        diffi_folder = os.path.join(folder, outlier_class, "diffi")
        
        for filename in os.listdir(shap_folder):
            if filename.endswith(".pkl"):
                idx = int(filename.split('_')[-1].split('.')[0])
                explanations[outlier_class][idx] = dict()
                explanations[outlier_class][idx]["shap"] = InteractionValues.load(os.path.join(shap_folder, filename))
        
        for filename in os.listdir(shap_iq_folder):
            if filename.endswith(".pkl"):
                idx = int(filename.split('_')[-1].split('.')[0])
                explanations[outlier_class][idx]["shap_iq"] = InteractionValues.load(os.path.join(shap_iq_folder, filename))
    
        for filename in os.listdir(diffi_folder):
            if filename.endswith(".npy"):
                idx = int(filename.split('_')[-1].split('.')[0])
                explanations[outlier_class][idx]["diffi"] = np.load(os.path.join(diffi_folder, filename))

    return explanations

# glass_explanations = load_glass_explanations()
# EXAMPLE: glass_explanations["out_5"][0]["shap"]

def load_glass_explanations_as_df():
    plot_path = "plots"
    CLASSES = ["out_5","out_6","out_7"]
    METHODS = ["shap","shap_iq"]
    path = "results/Glass_explanations/"
    explanation_dict = {}
    explanation_data = pd.DataFrame()

    #IMPORT EXPLANATIONS
    for output_class in CLASSES:
        class_path = path+output_class+"/"
        explanation_dict[output_class] = {}
        for method in METHODS:
                explanation_dict[output_class][method] = {}
                method_path = class_path+method+"/"
                # Loop through all files in the directory
                for file_name in os.listdir(method_path):
                    # Check if the file ends with .pkl (you can filter for specific file types if needed)
                    if file_name.endswith('.pkl'):
                        #SHAP, SHAP-IQ import
                        file_path = os.path.join(method_path, file_name)
                        id = file_name.split('_')[1][:-4]
                        # Load the interaction values using the method
                        explanation_dict[output_class][method][id] = InteractionValues.load_interaction_values(file_path)
                        data = pd.DataFrame(list(explanation_dict[output_class][method][id].dict_values.items()),columns=["interaction_subset","interaction_value"])
                        data["dataset"] = "Glass_explanations"
                        data["output_class"] = output_class
                        data["method"] = method
                        data["id"] = id
                        explanation_data = pd.concat([explanation_data,data])
                    if file_name.endswith('.npy'):
                        #DIFFI import
                        file_path = os.path.join(method_path, file_name)
                        id = file_name.split('_')[1][:-4]
                        values = np.load(file_path)
                        features = [(0,),(1,),(2,),(3,)]
                        # Load the interaction values using the method

                        data = pd.DataFrame(zip(features,values),columns=["interaction_subset","interaction_value"])
                        data["dataset"] = "Glass_explanations"
                        data["output_class"] = output_class
                        data["method"] = method
                        data["id"] = id
                        explanation_data = pd.concat([explanation_data,data])

    explanation_data = explanation_data[explanation_data["interaction_subset"] != ()]
    return explanation_data


def generate_explanations(clf, GlassData):
    folder = "results/Glass_explanations"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for outlier_class in ["out_5", "out_6", "out_7"]:
        for method in ["shap", "shap_iq"]:
            folder = f"results/Glass_explanations/{outlier_class}/{method}"
            if not os.path.exists(folder):
                os.makedirs(folder)

    # shap_isoforest_clf = convert_values_isolation_forest_shap_isotree(clf)
    shap_explainer = shapiq.TreeExplainer(model=clf, index="SV", max_order=1)
    shap_iq_explainer = shapiq.TreeExplainer(model=clf, index="k-SII", min_order=1, max_order=2)

    explanations = dict()
    # for each outlier_class create explanations
    for outlier_class in ["out_5", "out_6", "out_7"]:
        X_test, y_test = GlassData.test_sets[outlier_class]
        y_pred = clf.predict(X_test)
        # get the indices of the outliers
        outlier_indices = np.where(y_pred == -1)[0]

        # for each outlier create an explanation
        for idx in outlier_indices:
            print(f"Creating explanation for outlier {idx}")
            
            x = X_test[idx]
            explanations[idx] = dict()
            interaction_values_shap = shap_explainer.explain(x)
            interaction_values_shap_iq = shap_iq_explainer.explain(x)

            # save explanations to file
            print(f"Saving explanations for outlier {idx}", f"{folder}/{outlier_class}/shap/explanation_{idx}.pkl")
            interaction_values_shap.save(f"results/Glass_explanations/{outlier_class}/shap/explanation_{idx}.pkl")
            interaction_values_shap_iq.save(f"results/Glass_explanations/{outlier_class}/shap_iq/explanation_{idx}.pkl")


def generate_diffi_explanations(clf, GlassData):
    for outlier_class in ["out_5", "out_6", "out_7"]:
        X_test, y_test = GlassData.test_sets[outlier_class]
        y_pred = clf.predict(X_test)
        # get the indices of the outliers
        outlier_indices = np.where(y_pred == -1)[0]

        diffi_test, ord_idx_diffi_test, exec_time_diffi_test = local_diffi_batch(clf, X_test[np.where(y_pred == -1)])

        # extract indices of explanations
        expl_idxs = []
        folder = f"results/Glass_explanations/{outlier_class}/shap"
        for file in os.listdir(folder):
            expl_idxs.append(int(file.split("_")[1].split(".")[0]))

        folder = f"results/Glass_explanations/{outlier_class}/diffi"
        if not os.path.exists(folder):
            os.makedirs(folder)
        for diffi_test_idx, expl_idx in zip(diffi_test, expl_idxs):
            diffi_test_idx = diffi_test_idx.flatten()
            np.save(f"results/Glass_explanations/{outlier_class}/diffi/explanation_{expl_idx}.npy", diffi_test_idx)

def run_experiment():
    GlassData = load_glass_data()
    # Save data to file
    folder = 'results/Glass_explanations/train_test_data'
    if not os.path.exists(folder):
        os.makedirs(folder)

    GlassData.save_to_path(folder)
    # save test sets to numpy files
    for outlier_class in ["out_5", "out_6", "out_7"]:
        np.save(f"{folder}/X_test_{outlier_class}.npy", GlassData.test_sets[outlier_class][0])
        np.save(f"{folder}/y_test_{outlier_class}.npy", GlassData.test_sets[outlier_class][1])

    ## Generate model
    clf = IsolationForest(n_estimators=200, contamination='auto', random_state=None, bootstrap=False)
    clf.fit(GlassData.X_train)
    # Concatenate test sets for classes 5, 6, and 7
    X_test_5, y_test_5 = GlassData.test_sets["out_5"]
    X_test_6, y_test_6 = GlassData.test_sets["out_6"]
    X_test_7, y_test_7 = GlassData.test_sets["out_7"]
    X_test_concat = np.concatenate([X_test_5, X_test_6, X_test_7], axis=0)
    y_test_concat = np.concatenate([y_test_5, y_test_6, y_test_7], axis=0)

    # Predict on the concatenated test set
    y_pred_concat = clf.predict(X_test_concat)

    # Print outlier scores for the concatenated test set
    print_outlier_scores(y_pred_concat, y_test_concat)

    # Save the model
    # save clf to file
    folder = 'results/Glass_explanations/'
    with open(f"{folder}/isolation_forest_clf.pkl", "wb") as f:
        pickle.dump(clf, f)

    # save score to json
    accuracy, precision, recall, f1 = print_outlier_scores(y_pred_concat, y_test_concat)

    with open(f"{folder}/model_performance.json", "w") as f:
        json.dump({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}, f)

    # generate explanations
    generate_explanations(clf, GlassData)
    generate_diffi_explanations(clf, GlassData)

def plot_explanations(explanation_data):
    CLASSES = ["out_5","out_6","out_7"]
    METHODS = ["shap","shap_iq"]
    for outlier_class in CLASSES:
        for method in METHODS:
            plot_data = explanation_data[(explanation_data["output_class"]==outlier_class) & (explanation_data["method"]==method)]
            plot_data['interaction_subset'] = plot_data['interaction_subset'].astype(str)

            grouped = plot_data.groupby('interaction_subset')['interaction_value'].apply(list)

            grouped = grouped.sort_index()

            # Create a list of values for the violin plot
            data_to_plot = [group for group in grouped]
            # Plotting the violin plot
            # Create a figure
            plt.figure(figsize=(12, 6))

            # Create the violin plot
            plt.violinplot(data_to_plot)

            # Set the x-axis labels to the interaction subsets
            plt.xticks(np.arange(1, len(grouped) + 1), grouped.index, rotation=90)

            # Add labels and title
            plt.xlabel('Interaction Subset')
            plt.ylabel('Interaction Value')
            plt.title('Violin Plot of Interaction Values for ' + outlier_class + ' - ' + method)
            # Show the plot
            plt.tight_layout()  # To prevent clipping of labels
            plt.show()

if __name__ == "__main__":
    run_experiment()
    df = load_glass_explanations_as_df()
    plot_explanations(df)

