import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapiq.shapiq import InteractionValues

if __name__ == "__main__":
    plot_path = "plots/"
    DATASETS = ["diagonal_feature","single_outlier_quandrant"]
    TRIALS = ["trial_0","trial_1","trial_2","trial_3","trial_4","trial_5","trial_6","trial_7","trial_8","trial_9"]
    METHODS = ["diffi","shap","shap_iq"]
    path = "results/"
    explanation_dict = {}
    explanation_data = pd.DataFrame()

    #IMPORT EXPLANATIONS
    for dataset in DATASETS:
        dataset_path = path + dataset + "/"
        explanation_dict[dataset] = {}
        for trial in TRIALS:
            trial_path = dataset_path+trial+"/"
            explanation_dict[dataset][trial] = {}
            for method in METHODS:
                    explanation_dict[dataset][trial][method] = {}
                    method_path = trial_path+method+"/"
                    # Loop through all files in the directory
                    for file_name in os.listdir(method_path):
                        # Check if the file ends with .pkl (you can filter for specific file types if needed)
                        if file_name.endswith('.pkl'):
                            #SHAP, SHAP-IQ import
                            file_path = os.path.join(method_path, file_name)
                            id = file_name.split('_')[1][:-4]
                            # Load the interaction values using the method
                            explanation_dict[dataset][trial][method][id] = InteractionValues.load_interaction_values(file_path)
                            data = pd.DataFrame(list(explanation_dict[dataset][trial][method][id].dict_values.items()),columns=["interaction_subset","interaction_value"])
                            data["dataset"] = dataset
                            data["trial"] = trial
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
                            data["dataset"] = dataset
                            data["trial"] = trial
                            data["method"] = method
                            data["id"] = id
                            explanation_data = pd.concat([explanation_data,data])

    explanation_data = explanation_data[explanation_data["interaction_subset"] != ()]


    mins = explanation_data.groupby(["dataset","method","trial","id"])["interaction_value"].min().reset_index()
    maxs = explanation_data.groupby(["dataset","method","trial","id"])["interaction_value"].max().reset_index()
    sums = explanation_data.groupby(["dataset","method","trial","id"])["interaction_value"].sum().reset_index()
    metrics = pd.merge(mins,maxs,on=["dataset","method","trial","id"],how="inner",suffixes=("_min","_max"))
    explanation_data = pd.merge(explanation_data,mins,on=["dataset","method","trial","id"],how="left", suffixes=("","_min"))
    explanation_data = pd.merge(explanation_data,maxs,on=["dataset","method","trial","id"],how="left",suffixes=("","_max"))
    explanation_data = pd.merge(explanation_data,sums,on=["dataset","method","trial","id"],how="left", suffixes=("","_sum"))
    explanation_data["interaction_value_normalized"] = explanation_data["interaction_value"]/explanation_data["interaction_value_sum"]

    x1_x2_data = explanation_data[(explanation_data["interaction_subset"] == (0,))+(explanation_data["interaction_subset"] == (1,))+(explanation_data["interaction_subset"] == (0,1))]
    x1_x2_means = x1_x2_data.groupby(by=["dataset","method","interaction_subset"])["interaction_value_normalized"].mean()
    x1_x2_stds = x1_x2_data.groupby(by=["dataset","method","interaction_subset"])["interaction_value_normalized"].std()


    # Pivot the table for easier plotting
    x1_x2_means = x1_x2_data.groupby(by=["dataset","method","interaction_subset"])["interaction_value_normalized"].mean().reset_index()
    x1_x2_stds = x1_x2_data.groupby(by=["dataset","method","interaction_subset"])["interaction_value_normalized"].std().reset_index()


    #create the plots
    TITLES={"diffi":"DIFFI","shap":"SHAP","shap_iq":"Shapley Interactions (2-SVs)"}
    XLABELS=["Feature 1", "Feature 2", "Interaction"]
    CUSTOM_ORDER = [(0,), (1,), (0, 1)]

    x1_x2_means['interaction_subset'] = pd.Categorical(x1_x2_means['interaction_subset'],
                                                   categories=CUSTOM_ORDER, ordered=True)
    x1_x2_means = x1_x2_means.sort_values('interaction_subset')
    x1_x2_stds['interaction_subset'] = pd.Categorical(x1_x2_stds['interaction_subset'],
                                                       categories=CUSTOM_ORDER, ordered=True)
    x1_x2_stds = x1_x2_stds.sort_values('interaction_subset')

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4,5),sharex=True)


    for i,method in enumerate(METHODS):
        x1_x2_data = x1_x2_means[x1_x2_means["method"]==method]
        x1_x2_data_stds = x1_x2_stds[x1_x2_stds["method"]==method]

        x1_x2_data_pivot = x1_x2_data.pivot_table(index='interaction_subset',
                                                  columns=['dataset', 'method'],
                                                  values='interaction_value_normalized')
        x1_x2_data_pivot_stds = x1_x2_data_stds.pivot_table(index='interaction_subset',
                                                  columns=['dataset', 'method'],
                                                  values='interaction_value_normalized')
        x1_x2_data_pivot.plot(kind='bar', ax=axes[i], color=['skyblue', 'orange'],yerr=x1_x2_data_pivot_stds)
        axes[i].set_xlabel("")
        axes[i].set_title(TITLES[method],fontweight="bold")
        axes[i].legend().set_visible(False)
        if method in ["diffi","shap"]:
            axes[i].set_xticklabels(XLABELS[:2],rotation=0)
        else:
            axes[i].set_xticklabels(XLABELS,rotation=0)
        #plot legend

        axes[i].legend(["Simple Data","Interaction Data"],fontsize=6)

    # Adding labels and title
    axes[1].set_ylabel('Feature Importance')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(plot_path+"synthetic_data_comparison.eps")
    plt.savefig(plot_path+"synthetic_data_comparison.pdf")
    # Show plot
    plt.show()

    explanation_data["interaction_value_abs"] = np.abs(
        explanation_data["interaction_value_normalized"])

    abs_means = explanation_data.groupby(by=["dataset", "method", "interaction_subset"])[
        "interaction_value_abs"].mean()
    abs_stds = explanation_data.groupby(by=["dataset", "method", "interaction_subset"])[
        "interaction_value_abs"].std()
    stds = explanation_data.groupby(by=["dataset", "method", "interaction_subset"])[
        "interaction_value_abs"].std()

    for dataset in DATASETS:
        for method in METHODS:
            plot_data = explanation_data[(explanation_data["dataset"]==dataset) & (explanation_data["method"]==method)]
            plot_data['interaction_subset'] = plot_data['interaction_subset'].astype(str)

            grouped = plot_data.groupby('interaction_subset')['interaction_value_normalized'].apply(list)

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
            plt.ylabel('Interaction Value')
            plt.title('Violin Plot of Interaction Values for ' + dataset + ' - ' + method)
            # Show the plot
            plt.tight_layout()  # To prevent clipping of labels
            plt.show()


    max_per_id = explanation_data.groupby(by=["dataset","trial","method","id"])["interaction_value_abs"].max()
    max_per_id = pd.merge(explanation_data,max_per_id,on=["dataset","trial","method","id","interaction_value_abs"])

    min_per_id = explanation_data.groupby(by=["dataset","trial","method","id"])["interaction_value_normalized"].min()
    min_per_id = pd.merge(explanation_data,min_per_id,on=["dataset","trial","method","id","interaction_value_normalized"])

    means_ = explanation_data.groupby(by=["dataset","trial","method","interaction_subset"])["interaction_value_normalized"].mean()

    count_min = max_per_id.groupby(by=["dataset","method","interaction_subset"])["id"].count()
    count_max = max_per_id.groupby(by=["dataset","method","interaction_subset"])["id"].count()

    top_k_per_id = explanation_data.groupby(by=["dataset","trial","method","id"])["interaction_value_abs"].nlargest(3)
    top_k_per_id = pd.merge(explanation_data,top_k_per_id,on=["dataset","trial","method","id","interaction_value_abs"])
    count_top_k = top_k_per_id.groupby(by=["dataset","method","interaction_subset"])["id"].count()

