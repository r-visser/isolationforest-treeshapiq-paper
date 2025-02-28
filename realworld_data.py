import pickle
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from helpers import convert_labels_to_isoforest
from synthetic_data import add_empty_interactions
from trials import OutlierSet, TrialData


def load_glass_data(outliers:Literal['all', 'out_7']= 'all'):
    # Load pickle
    with open('data/Glass/glass.pkl', 'rb') as f:
        glass = pickle.load(f)

    random_state = 0

    x_in = glass['X_in']
    y_in = convert_labels_to_isoforest(glass['y_in'])
    x_out_5 = glass['X_out_5']
    y_out_5 = convert_labels_to_isoforest(glass['y_out_5'])
    x_out_6 = glass['X_out_6']
    y_out_6 = convert_labels_to_isoforest(glass['y_out_6'])
    x_out_7 = glass['X_out_7']
    y_out_7 = convert_labels_to_isoforest(glass['y_out_7'])

    if outliers == 'all':
        # sample 5,6,7
        test_size = 0.7
        X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(x_out_5, y_out_5, test_size=test_size, random_state=random_state)
        X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(x_out_6, y_out_6, test_size=test_size, random_state=random_state)
        X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(x_out_7, y_out_7, test_size=test_size, random_state=random_state)
        
        # training data
        X_train = np.concatenate((x_in, X_train_5, X_train_6, X_train_7))
        y_train = np.concatenate((y_in, y_train_5, y_train_6, y_train_7))
        
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
        # test outliers
        X_test = np.concatenate((X_test_5, X_test_6, X_test_7))
        y_test = np.concatenate((y_test_5, y_test_6, y_test_7))
    elif outliers == 'out_7':
        # training data
        X_train = np.concatenate((x_in, x_out_5, x_out_6))
        y_train = np.concatenate((y_in, y_out_5, y_out_6))
        
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
        # test outliers
        X_test = x_out_7
        y_test = y_out_7

    print("size inliers:", glass['X_in'].shape)
    print("size outliers 5:", glass['X_out_5'].shape)
    print("size outliers 6:", glass['X_out_6'].shape)
    print("size outliers 7:", glass['X_out_7'].shape)

    # create labels "cl5", "cl6", "cl7" for groups in test set
    class_labels = []
    class_labels.extend([f"out_7" for _ in range(len(glass['X_out_7']))])
    class_labels.extend([f"out_5" for _ in range(len(glass['X_out_5']))])
    class_labels.extend([f"out_6" for _ in range(len(glass['X_out_6']))])
    class_labels = np.array(class_labels)


    id2feat = {0:'RI', 1:'Na', 2:'Mg', 3:'Al', 4:'Si', 5:'K', 6:'Ca', 7:'Ba', 8:'Fe'}
    feature_labels=list(id2feat.values())
    ground_truth = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    ground_truth_outlier_features_test = np.array([ground_truth for _ in range(X_test.shape[0])])
    # add zero array for inliers
    ground_truth_outlier_features = np.concatenate((np.zeros((X_train.shape[0], 9)), ground_truth_outlier_features_test), axis=0)

    ground_truth_outlier_features = add_empty_interactions(ground_truth_outlier_features)
    ground_truth_outlier_features_test = add_empty_interactions(ground_truth_outlier_features_test)


    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    outlier_set = OutlierSet("GlassOutliers", idx=np.where(y == -1)[0], outlier_feature_labels=['Al', 'Ba'])

    GlassData = TrialData(
        X=X,
        y=y,
        feature_labels=feature_labels,
        # ground_truth_outlier_features=ground_truth_outlier_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        ground_truth_outlier_features_test=ground_truth_outlier_features_test,
        outlier_sets=[outlier_set]
    )
    GlassData.class_labels = class_labels
    if outliers == 'all':
        GlassData.test_sets = {'out_5': (X_test_5, y_test_5), 'out_6': (X_test_6, y_test_6), 'out_7': (X_test_7, y_test_7)}
    elif outliers == 'out_7':
        GlassData.test_sets = {'out_7': (X_test, y_test)}
    return GlassData


def load_pulsar_data():
    column_names = ['Profile_mean', 'Profile_stdev', 'Profile_skewness', 'Profile_kurtosis', 'DM_mean', 'DM_stdev', 'DM_skewness', 'DM_kurtosis', 'class']

    # load pulsar csv
    pulsar_df = pd.read_csv('data/Pulsar/htru2/HTRU_2.csv', names=column_names)

    # convert df.class using convert_labels_to_isoforest
    from helpers import convert_labels_to_isoforest

    pulsar_df['class'] = convert_labels_to_isoforest(pulsar_df['class'])

    # take all class 1 as inliers and subset of class -1 as outliers
    pulsar_df_inliers = pulsar_df[pulsar_df['class'] == 1]
    pulsar_df_outliers = pulsar_df[pulsar_df['class'] == -1].sample(200)

    # Create X, y, feature_labels, outlier_sets and create TrialData object
    X = pd.concat([pulsar_df_inliers, pulsar_df_outliers]).drop(columns=['class']).to_numpy()
    y = pd.concat([pulsar_df_inliers, pulsar_df_outliers])['class'].to_numpy()
    feature_labels = column_names[:-1]
    outlier_set = OutlierSet("PulsarOutliers", idx=np.where(y == -1)[0], outlier_feature_labels=['Profile_skewness', 'Profile_kurtosis'])
    pulsar_data = TrialData(X, y, feature_labels, [outlier_set])
    return pulsar_data