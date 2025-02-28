import matplotlib.pyplot as plt

from synthetic_data import (SingleFeatureDiagonalGenerator,
                            SingleOutlierQuadrantGenerator)
from trials import TrialData


def plot_synethtic_data(data: TrialData, correlated_features=[0, 1],color="skyblue",plot_path="",title=""):
    inliers = data.inliers
    outliers = data.outliers

    fontsize = 30
    plt.figure(figsize=(8, 7.5))

    # Select only the correlated features for both inliers and outliers
    inliers_correlated = inliers[:, correlated_features]
    outliers_correlated = outliers[:, correlated_features]

    # Plot inliers in blue
    plt.scatter(inliers_correlated[:, 0], inliers_correlated[:, 1], color=color, label='Inliers',marker="o",
                alpha=0.5)

    # Plot outliers in red
    plt.scatter(outliers_correlated[:, 0], outliers_correlated[:, 1], color=color, label='Outliers',marker="x",s=200,
                alpha=1)

    if color == "orange":
        plt.gca().yaxis.set_label_position("right")
        plt.gca().yaxis.tick_right()

    plt.title(title,fontsize=fontsize, fontweight='bold')
    plt.xlabel(f'Feature {correlated_features[0] + 1}', fontsize=fontsize,fontweight='bold')
    plt.ylabel(f'Feature {correlated_features[1] + 1}',fontsize=fontsize, fontweight='bold')
    plt.legend(fontsize=fontsize,loc='upper right')
    #plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    plot_path = "plots/"

    SFGenerator = SingleFeatureDiagonalGenerator(noise_level=0.1)
    SimpleData = SFGenerator.generate_inliers_outliers(n_inliers=2500, n_outliers=50, n_random_features=2)

    SOGenerator = SingleOutlierQuadrantGenerator(noise_level=0.1, outlier_location=[0, 1])
    InteractionData = SOGenerator.generate_inliers_outliers(n_inliers=2500, n_outliers=50,
                                                   n_random_features=2)

    plot_synethtic_data(SimpleData, [0, 1],color="skyblue",plot_path=plot_path + "simple_data.eps",title="Simple Data")
    plot_synethtic_data(SimpleData, [0, 1],color="skyblue",plot_path=plot_path + "simple_data.pdf",title="Simple Data")
    plot_synethtic_data(InteractionData, [0, 1],color="orange",plot_path=plot_path+"interaction_data.eps",title="Interaction Data")
    plot_synethtic_data(InteractionData, [0, 1],color="orange",plot_path=plot_path+"interaction_data.pdf",title="Interaction Data")


    import matplotlib.pyplot as plt
    import numpy as np

    #visualize only training data 20% of 2500 = 500
    #TODO: verify if noise level coincides with experimental setup
    data1 = [1,0] + 0.1*np.random.randn(500, 2)
    data2 = [0,1] + 0.1*np.random.randn(500,2)

    plt.figure(figsize=(8, 7.5))
    plt.scatter(x=data1[:,0],y=data1[:,1])
    plt.scatter(x=data2[:,0],y=data2[:,1])
    plt.show()