import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapiq.shapiq import InteractionValues
from shapiq.shapiq.plot import (force_plot, si_graph_plot,
                                stacked_bar_plot)

if __name__ == "__main__":
    plot_path = "plots/glass/"
    CLASSES = ["out_5","out_6","out_7"]
    METHODS = ["shap","shap_iq"]
    path = "results/Glass_explanations/"
    explanation_dict = {}
    explanation_data = pd.DataFrame()

    id2feat = {0: 'RI', 1: 'Na', 2: 'Mg', 3: 'Al', 4: 'Si', 5: 'K', 6: 'Ca', 7: 'Ba', 8: 'Fe'}
    feature_labels = list(id2feat.values())

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



    mins = explanation_data.groupby(["dataset","method","output_class","id"])["interaction_value"].min().reset_index()
    maxs = explanation_data.groupby(["dataset","method","output_class","id"])["interaction_value"].max().reset_index()
    sums = explanation_data.groupby(["dataset","method","output_class","id"])["interaction_value"].sum().reset_index()
    metrics = pd.merge(mins,maxs,on=["dataset","method","output_class","id"],how="inner",suffixes=("_min","_max"))
    explanation_data = pd.merge(explanation_data,mins,on=["dataset","method","output_class","id"],how="left", suffixes=("","_min"))
    explanation_data = pd.merge(explanation_data,maxs,on=["dataset","method","output_class","id"],how="left",suffixes=("","_max"))
    explanation_data = pd.merge(explanation_data,sums,on=["dataset","method","output_class","id"],how="left", suffixes=("","_sum"))
    explanation_data["interaction_value_normalized"] = explanation_data["interaction_value"]/explanation_data["interaction_value_sum"]



    explanation_data["interaction_value_abs"] = np.abs(
        explanation_data["interaction_value_normalized"])

    abs_means = explanation_data.groupby(by=["dataset", "method", "interaction_subset"])[
        "interaction_value_abs"].mean()
    abs_stds = explanation_data.groupby(by=["dataset", "method", "interaction_subset"])[
        "interaction_value_abs"].std()
    stds = explanation_data.groupby(by=["dataset", "method", "interaction_subset"])[
        "interaction_value_abs"].std()

    for output_class in CLASSES:
        for method in METHODS:
            plot_data = explanation_data[(explanation_data["output_class"]==output_class)*(explanation_data["method"]==method)]
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
            plt.title('Violin Plot of Interaction Values for glass class ' + output_class + ' - ' + method)
            # Show the plot
            plt.tight_layout()  # To prevent clipping of labels
            plt.show()


    n_interactions = 11

    for output_class in CLASSES:
        ids = explanation_dict[output_class]["shap"].keys()
        for id in ids:
            for method in METHODS:
                si = explanation_dict[output_class][method][id]
                #reset baseline for plotting
                sum = si.values.sum()-si[tuple()]
                si.values /= sum
                plt.figure()
                force_plot(si,show=False,feature_names=feature_labels)
                plt.savefig(plot_path+output_class+"/force_"+ output_class + "_"+method+"_"+id+".eps")
                plt.figure()
                fig = stacked_bar_plot(explanation_dict[output_class][method][id],feature_names=feature_labels,show=False)
                plt.savefig(plot_path+output_class+"/stackedbars_"+ output_class + "_"+method+"_"+id+".eps")
                plt.figure()
                si_graph_plot(si,feature_names=feature_labels,show=False,n_interactions=n_interactions,size_factor=3)
                plt.savefig(plot_path+output_class+"/sigraph_"+ output_class + "_"+method+"_"+id+".eps")

    for output_class in CLASSES:
        for method in METHODS:
            ids = explanation_dict[output_class][method].keys()
            for i,id in enumerate(ids):
                si_current = explanation_dict[output_class][method][id]
                sum = si_current.values.sum()-si_current[tuple()]
                si_current.values /= sum
                #si_current.values = np.abs(si_current.values)
                if i == 0:
                    average_si = si_current
                else:
                    average_si += si_current

            average_si = average_si* (1/len(ids))


        
            plt.figure()
            fig=force_plot(average_si, show=False, feature_names=feature_labels)
            plt.tight_layout()
            plt.savefig(plot_path +  "averages/" + "/force_" + output_class + "_" + method +".eps")

            plt.figure()
            fig = stacked_bar_plot(average_si, feature_names=feature_labels, show=False)
            plt.tight_layout()
            plt.savefig(plot_path + "averages/" + "/stackedbars_" + output_class + "_"+ method + ".eps")
            plt.figure()
            si_graph_plot(average_si, feature_names=feature_labels, n_interactions=n_interactions,show=False,size_factor=5)
            plt.tight_layout()
            plt.savefig(plot_path + "averages/" + "/sigraph_" + output_class + "_" + method + ".eps")

