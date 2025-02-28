import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def calculate_outlier_scores(y_pred, y_true):
    """
    Calculate the outlier scores and the performance metrics of the isolation forest.
    """
    # Convert 1 for inliers and -1 for outliers of IsoForest to 0 for inliers and 1 for outliers
    y_pred_binary = (y_pred == -1).astype(int)
    y_true_binary = (y_true == -1).astype(int)

    # Compute accuracy
    accuracy = np.mean(y_pred_binary == y_true_binary)

    # Compute precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average="binary")

    return accuracy, precision, recall, f1

def print_outlier_scores(y_pred, y_true):
    """
    Print the outlier scores and the performance of the isolation forest.
    """
    accuracy, precision, recall, f1 = calculate_outlier_scores(y_pred, y_true)
    print("The performance of the isoforest to detect outliers:")
    print("Accuracy:", accuracy)
    print("Precision:", precision, ' -- (amount of detected outliers which are actual outliers)')
    print("Recall:", recall, ' -- (amount of actual outliers which are detected as outliers)')
    print("F1 Score:", f1)

    return accuracy, precision, recall, f1

def convert_labels_to_isoforest(y):
    # convert 1 to -1 and 0 to 1
    return np.where(y == 1, -1, 1)

def create_interaction_labels(feature_labels):
    interactions = []
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            if i < j:
                interactions.append(f"({feature1}, {feature2})")
    return interactions