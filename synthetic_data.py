from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from helpers import create_interaction_labels
from trials import OutlierSet, TrialData


@dataclass
class SyntheticOutlierInlierSettings:
    n_samples: int = 1000
    n_outliers: int = 100
    n_clusters: int = 2
    n_features: int = 12
    n_perturb_features: int = 2
    always_perturb_same_features: bool = True
    noise_level: float = 5
    random_state: int = None

@dataclass
class SyntheticOutlierInlierData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    original_samples: np.ndarray
    perturbed_samples: np.ndarray
    original_to_perturbed: dict
    settings: SyntheticOutlierInlierSettings
    ground_truth_outlier_features: np.ndarray


def find_perturbed_features(synthetic_data: SyntheticOutlierInlierData):
    settings = synthetic_data.settings
    # print an array of which features are perturbed
    perturbed_features = np.zeros((settings.n_outliers, settings.n_features))
    for i in range(settings.n_outliers):
        perturbed_features[i, np.where(synthetic_data.original_samples[i] != synthetic_data.perturbed_samples[i])] = 1
    print("Perturbed Features:\n", perturbed_features)


def visualize_interaction(data: TrialData, correlated_features=[0, 1]):
    inliers = data.inliers
    outliers = data.outliers
    
    plt.figure(figsize=(8, 6))
    
    # Select only the correlated features for both inliers and outliers
    inliers_correlated = inliers[:, correlated_features]
    outliers_correlated = outliers[:, correlated_features]
    
    # Plot inliers in blue
    plt.scatter(inliers_correlated[:, 0], inliers_correlated[:, 1], color='blue', label='Inliers', alpha=0.6)
    
    # Plot outliers in red
    plt.scatter(outliers_correlated[:, 0], outliers_correlated[:, 1], color='red', label='Outliers', alpha=0.8)
    
    plt.title('2D Visualization of Correlated Features')
    plt.xlabel(f'Feature {correlated_features[0] + 1}')
    plt.ylabel(f'Feature {correlated_features[1] + 1}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return plt


def visualize_2d_correlated(inliers, outliers, correlated_features=[0, 1]):
    plt.figure(figsize=(8, 6))
    
    # Select only the correlated features for both inliers and outliers
    inliers_correlated = inliers[:, correlated_features]
    outliers_correlated = outliers[:, correlated_features]
    
    # Plot inliers in blue
    plt.scatter(inliers_correlated[:, 0], inliers_correlated[:, 1], color='blue', label='Inliers', alpha=0.6)
    
    # Plot outliers in red
    plt.scatter(outliers_correlated[:, 0], outliers_correlated[:, 1], color='red', label='Outliers', alpha=0.8)
    
    plt.title('2D Visualization of Correlated Features')
    plt.xlabel(f'Feature {correlated_features[0] + 1}')
    plt.ylabel(f'Feature {correlated_features[1] + 1}')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_3d_correlated(inliers, outliers, correlated_features=[0, 1, 2]):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select only the correlated features for both inliers and outliers
    inliers_correlated = inliers[:, correlated_features]
    outliers_correlated = outliers[:, correlated_features]
    
    # Plot inliers in blue
    ax.scatter(inliers_correlated[:, 0], inliers_correlated[:, 1], inliers_correlated[:, 2], color='blue', label='Inliers', alpha=0.6)
    
    # Plot outliers in red
    ax.scatter(outliers_correlated[:, 0], outliers_correlated[:, 1], outliers_correlated[:, 2], color='red', label='Outliers', alpha=0.8)
    
    ax.set_title('3D Visualization of Correlated Features')
    ax.set_xlabel(f'Feature {correlated_features[0] + 1}')
    ax.set_ylabel(f'Feature {correlated_features[1] + 1}')
    ax.set_zlabel(f'Feature {correlated_features[2] + 1}')
    ax.legend()
    plt.show()

def orthogonal_vector(v):
    """Generate an orthogonal vector to the given vector v in N-dimensional space."""
    v_orth = np.random.rand(len(v))
    v_orth -= v_orth.dot(v) / np.linalg.norm(v)**2 * v  # Project orthogonal
    return v_orth / np.linalg.norm(v_orth)  # Normalize the orthogonal vector

def distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(a - b)

def get_correlation_direction(inliers, features):
    """Calculate the direction vector of the correlated features in the dataset."""
    return inliers[:, features].mean(axis=0)

def generate_data_with_orthogonal_perturbation(N=5, n_inliers=1000, n_outliers=50, 
                  correlation_strength=0.8, perturbation_magnitude=1.5, 
                  min_perturbation=1.0, add_noise=True, noise_level=0.1):
    """
    Generates an N-dimensional dataset with inliers and perturbed outliers along orthogonal directions.
    
    Returns:
    - inliers: Generated inlier data.
    - outliers: Generated outlier data.
    - labels: Labels for inliers (0) and outliers (1).
    - ground_truth: Ground truth of which features were perturbed for each outlier.
    """
    mean_inliers = np.zeros(N)
    cov_inliers = np.eye(N)
    
    # Define the correlated features
    perturb_features = [0, 1]
    
    # Set correlation strength between the correlated features
    cov_inliers[perturb_features[0], perturb_features[1]] = correlation_strength
    cov_inliers[perturb_features[1], perturb_features[0]] = correlation_strength

    # Generate inliers
    inliers = np.random.multivariate_normal(mean_inliers, cov_inliers, size=n_inliers)

    # Compute feature bounds for clipping
    feature_bounds = np.array([inliers.min(axis=0), inliers.max(axis=0)])

    # Generate outliers based on inliers
    outliers = np.copy(inliers[:n_outliers])

    # Get the correlation direction
    correlation_direction = get_correlation_direction(inliers, perturb_features)

    for i in range(n_outliers):
        while True:  # Ensure we meet the minimum perturbation distance
            # Calculate the direction vector to perturb (orthogonal to the correlation direction)
            direction = correlation_direction - mean_inliers[perturb_features]
            orthogonal_dir = orthogonal_vector(direction)  # Get an orthogonal vector
            
            # Generate a random perturbation magnitude along the orthogonal direction
            perturbation = np.random.uniform(min_perturbation, perturbation_magnitude)
            
            # Apply perturbation to correlated features along the orthogonal direction
            outliers[i, perturb_features] += orthogonal_dir * perturbation
            
            # Ensure values stay within bounds of inliers
            outliers[i] = np.clip(outliers[i], feature_bounds[0], feature_bounds[1])

            # Check the distance from the corresponding inlier
            if distance(outliers[i], inliers[i]) >= min_perturbation:
                break  # Exit loop if minimum distance is satisfied

    # Add Gaussian noise to uncorrelated features if specified
    if add_noise:
        uncorrelated_features = [i for i in range(N) if i not in perturb_features]
        noise = np.random.normal(0, noise_level, size=(n_outliers, len(uncorrelated_features)))
        outliers[:, uncorrelated_features] += noise

    # Create labels for the dataset (0 for inliers, 1 for outliers)
    labels = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)])

    # Ground truth for perturbed features (1 indicates a feature was perturbed)
    ground_truth = np.zeros((n_outliers, N))
    ground_truth[:, perturb_features] = 1

    # TODO remove original inliers used to create outliers from inliers?

    return inliers, outliers, labels, ground_truth

def add_empty_interactions(ground_truth_outlier_features):
    # add as many columns as the combination of features
    n_features = ground_truth_outlier_features.shape[1]
    n_combinations = len([(i, j) for i in range(n_features) for j in range(i + 1, n_features)])
    ground_truth_outlier_features = np.concatenate([ground_truth_outlier_features, np.zeros((ground_truth_outlier_features.shape[0], n_combinations))], axis=1)
    return ground_truth_outlier_features

def add_feature_combinations(y, ground_truth_outlier_features, feature_labels):
    n_inliers = np.sum(y == 1)
    n_outliers = np.sum(y == -1)
    # add combination of features
    outlier_features_first_order = ground_truth_outlier_features
    # create tuples for each combination of features
    outlier_features_ground_truth = [(i, j) for i in range(X.shape[1]) for j in range(i+1, X.shape[1])]
    feature_labels += [f'{t}' for t in outlier_features_ground_truth]
    
    interaction_features = [(0,1)]
    outlier_features_y_second_order = [1 if i in interaction_features else 0 for i in outlier_features_ground_truth]
    ground_truth_outlier_features = np.tile(np.concatenate([outlier_features_first_order, outlier_features_y_second_order]), (n_outliers, 1)) # TODO fix lenght of y
    # create n_inliers rows of zeros
    ground_truth_inlier_features = np.zeros((n_inliers, ground_truth_outlier_features.shape[1]))
    ground_truth_outlier_features = np.concatenate([ground_truth_inlier_features, ground_truth_outlier_features])
    return ground_truth_outlier_features, feature_labels


class DataGenerator(ABC):
    @abstractmethod
    def generate_inliers_outliers(self, n_inliers=1000, n_outliers=50, n_random_features=2):
        pass

    @abstractmethod
    def generate_inliers(self, n_inliers=100):
        pass

    @abstractmethod
    def generate_outliers(self, n_outliers=20):
        pass


class CorrelatedFeaturesGenerator(DataGenerator):
    def __init__(self, correlation_strength=0.95, perturbation_magnitude=3.0, min_perturbation=1.5, add_noise=True, noise_level=0.1):
        self.correlation_strength = correlation_strength
        self.perturbation_magnitude = perturbation_magnitude
        self.min_perturbation = min_perturbation
        self.add_noise = add_noise
        self.noise_level = noise_level

    def generate_inliers_outliers(self, n_inliers=1000, n_outliers=50, n_random_features=2):
        inliers, outliers, labels, ground_truth = generate_data_with_orthogonal_perturbation(
            N=n_random_features+2, 
            n_inliers=n_inliers, 
            n_outliers=n_outliers, 
            correlation_strength=self.correlation_strength, 
            perturbation_magnitude=self.perturbation_magnitude, 
            min_perturbation=self.min_perturbation, 
            add_noise=self.add_noise, 
            noise_level=self.noise_level
        )

        X = np.concatenate((inliers, outliers), axis=0)

        # change y labels to -1 for outliers
        y = np.copy(labels)
        y[y == 1] = -1
        y[y == 0] = 1

        # TODO remove duplication from this and generate_inliers_outliers:
        feature_labels = ['Cor_0', 'Cor_1'] + [f'Rnd_{i}' for i in range(n_random_features)]
        # add combination of features
        feature_labels += create_interaction_labels(feature_labels)


        outlier_set = OutlierSet("CorrelatedFeatures", idx=np.where(y == -1)[0], outlier_feature_labels=['Cor_0', 'Cor_1', '(Cor_0, Cor_1)'])
        data = TrialData(X, y, feature_labels, outlier_sets=[outlier_set])
        return data
    
    def generate_inliers(self, n_inliers=1000):
        inliers, outliers, labels, ground_truth = generate_data_with_orthogonal_perturbation(
            N=2, 
            n_inliers=n_inliers, 
            n_outliers=0, 
            correlation_strength=self.correlation_strength, 
            perturbation_magnitude=self.perturbation_magnitude, 
            min_perturbation=self.min_perturbation, 
            add_noise=self.add_noise, 
            noise_level=self.noise_level
        )
        return inliers, np.ones(n_inliers)
    
    def generate_outliers(self, n_outliers=50):
        inliers, outliers, labels, ground_truth = generate_data_with_orthogonal_perturbation(
            N=2, 
            n_inliers=n_outliers, 
            n_outliers=n_outliers, 
            correlation_strength=self.correlation_strength, 
            perturbation_magnitude=self.perturbation_magnitude, 
            min_perturbation=self.min_perturbation, 
            add_noise=self.add_noise, 
            noise_level=self.noise_level
        )
        return outliers, np.ones(n_outliers)*-1


class XORGenerator(DataGenerator):
    """Generate synthetic data (features) with XOR pattern interaction, 
        where the outliers are on one diagonal (i.e. top-left, bottom-right quadrants) and inliers on the other diagonal (i.e. bottom-left, top-right quadrants).
    """
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
    
    def generate_inliers_outliers(self, n_inliers=1000, n_outliers=20, n_random_features=5):
        X_inliers, y_inliers = self.generate_inliers(n_inliers)
        X_outliers, y_outliers = self.generate_outliers(n_outliers)

        X = np.concatenate([X_inliers, X_outliers])
        y = np.concatenate([y_inliers, y_outliers])

        # add random noise features
        X = np.column_stack([X, np.random.randn(X.shape[0], n_random_features)])

        feature_labels = ['XOR_0', 'XOR_1'] + [f'Rnd_{i}' for i in range(n_random_features)]
        # create tuples for each combination of features
        feature_labels += create_interaction_labels(feature_labels)

        outlier_set = OutlierSet("XOR", idx=np.where(y == -1)[0], outlier_feature_labels=['XOR_0', 'XOR_1', "(XOR_0, XOR_1)"])
        data = TrialData(X, y, feature_labels, outlier_sets=[outlier_set])
        return data
    
    def generate_single_feature_inliers_outliers(self, n_inliers=1000, n_outliers=20, direction:Literal["horizontal", "vertical"]="horizontal"):
        if direction == "horizontal":
            X_inliers_0, y_inliers_0 = self._generate_data(0, 0, n_inliers // 2, label=1)
            X_inliers_1, y_inliers_1 = self._generate_data(1, 0, n_inliers // 2, label=1)
            X_inliers = np.concatenate([X_inliers_0, X_inliers_1])
            y_inliers = np.concatenate([y_inliers_0, y_inliers_1])

            X_outliers_0, y_outliers_0 = self._generate_data(0, 1, n_outliers // 2, label=-1)
            X_outliers_1, y_outliers_1 = self._generate_data(1, 1, n_outliers // 2, label=-1)
            X_outliers = np.concatenate([X_outliers_0, X_outliers_1])
            y_outliers = np.concatenate([y_outliers_0, y_outliers_1])
        else:
            X_inliers_0, y_inliers_0 = self._generate_data(0, 0, n_inliers // 2, label=1)
            X_inliers_1, y_inliers_1 = self._generate_data(0, 1, n_inliers // 2, label=1)
            X_inliers = np.concatenate([X_inliers_0, X_inliers_1])
            y_inliers = np.concatenate([y_inliers_0, y_inliers_1])

            X_outliers_0, y_outliers_0 = self._generate_data(1, 1, n_outliers // 2, label=-1)
            X_outliers_1, y_outliers_1 = self._generate_data(1, 0, n_outliers // 2, label=-1)
            X_outliers = np.concatenate([X_outliers_0, X_outliers_1])
            y_outliers = np.concatenate([y_outliers_0, y_outliers_1])

        X = np.concatenate([X_inliers, X_outliers])
        y = np.concatenate([y_inliers, y_outliers])

        feature_labels = ['XOR_0', 'XOR_1']
        # create tuples for each combination of features
        feature_labels += create_interaction_labels(feature_labels)

        if direction == "horizontal":
            outlier_feature_labels = ['XOR_1']
        else:
            outlier_feature_labels = ['XOR_0']
        outlier_set = OutlierSet("XOR", idx=np.where(y == -1)[0], outlier_feature_labels=outlier_feature_labels)
        data = TrialData(X, y, feature_labels, outlier_sets=[outlier_set])
        return data
    
    def generate_inliers(self, n_inliers=100):
        """Generate data features for the inlier class, points at (0,0) and (1,1)

        Returns:
            X_inliers, y_inliers: inlier data features and labels (1 for inliers)
        """
        # 
        X1_inliers = np.concatenate([np.zeros(n_inliers // 2), np.ones(n_inliers // 2)])
        X2_inliers = X1_inliers  # Matching diagonal points
        y_inliers = np.ones(n_inliers)  # Class 0 for inliers
        X1_inliers_noisy = X1_inliers + self.noise_level * np.random.randn(n_inliers)
        X2_inliers_noisy = X2_inliers + self.noise_level * np.random.randn(n_inliers)
        return np.column_stack([X1_inliers_noisy, X2_inliers_noisy]), y_inliers
    
    def generate_outliers(self, n_outliers=20):
        """Generate data features for the outlier class, points at (0,1) and (1,0)

        Returns:
            X_outliers, y_outliers: outlier data features and labels (-1 for outliers)
        """
        # Generate data for the outlier class (class 1), points at (0,1) and (1,0)
        X1_outliers = np.concatenate([np.zeros(n_outliers // 2), np.ones(n_outliers // 2)]) # (0,1) outliers
        X2_outliers = 1 - X1_outliers  # Opposite diagonal points - (1,0) outliers
        y_outliers = np.ones(n_outliers)*-1  # Class 1 for outliers
        X1_outliers_noisy = X1_outliers + self.noise_level * np.random.randn(n_outliers)
        X2_outliers_noisy = X2_outliers + self.noise_level * np.random.randn(n_outliers)
        y_causal = ["0,1"] * len(X1_outliers_noisy)
        y_causal += ["1,0"] * len(X2_outliers_noisy)
        return np.column_stack([X1_outliers_noisy, X2_outliers_noisy]), y_outliers
    
    def _generate_data(self, x_value, y_value, n_samples=100, label=1):
        data = np.column_stack([np.full(n_samples, x_value), np.full(n_samples, y_value)])
        data_noisy = data + self.noise_level * np.random.randn(n_samples, 2)
        labels = np.full(n_samples, label)
        return data_noisy, labels

    def generate_0_1_outliers(self, n_outliers=20):
        return self._generate_data(0, 1, n_outliers, label=-1)

    def generate_1_0_outliers(self, n_outliers=20):
        return self._generate_data(1, 0, n_outliers, label=-1)

    def generate_0_0_inliers(self, n_inliers=100):
        return self._generate_data(0, 0, n_inliers, label=1)

    def generate_1_1_inliers(self, n_inliers=100):
        return self._generate_data(1, 1, n_inliers, label=1)
    

class SingleFeatureXORGenerator(XORGenerator):
    def __init__(self, noise_level=0.1, direction:Literal["horizontal", "vertical"]="horizontal"):
        self.noise_level = noise_level
        self.direction = direction

    @property
    def outlier_features_idx(self) -> list[int]:
        if self.direction == "horizontal":
            return [0]
        else:
            return [1]

    def generate_inliers(self, n_inliers=100):
        return self.generate_0_0_inliers(n_inliers) # TODO fix outlier_feature_labels if used for add_outliers trialData
    
    def generate_outliers(self, n_outliers=20):
        if self.direction == "horizontal":
            return self.generate_1_0_outliers(n_outliers)
        else:
            return self.generate_0_1_outliers(n_outliers)
        
class SingleFeatureDiagonalGenerator(XORGenerator):
    def __init__(self, noise_level=0.05, inlier_location=[1, 0]):
        super().__init__(noise_level)
        self.inlier_location = inlier_location

    def generate_inliers_outliers(self, n_inliers=1000, n_outliers=20, n_random_features=5):
        X_inliers, y_inliers = self._generate_data(self.inlier_location[0], self.inlier_location[1], n_inliers, label=1)
        X_outliers, y_outliers = self._generate_data(1 - self.inlier_location[0], 1 - self.inlier_location[1], n_outliers, label=-1)

        X = np.concatenate([X_inliers, X_outliers])
        y = np.concatenate([y_inliers, y_outliers])

        # add random noise features
        X = np.column_stack([X, np.random.randn(X.shape[0], n_random_features)])

        feature_labels = ['XOR_0', 'XOR_1'] + [f'Rnd_{i}' for i in range(n_random_features)]
        # create tuples for each combination of features
        feature_labels += create_interaction_labels(feature_labels)

        outlier_set = OutlierSet("XOR_singlefeature_diagonal", idx=np.where(y == -1)[0], outlier_feature_labels=['XOR_0', 'XOR_1']) # TODO discuss: add interaction features or not? -> , '(XOR_0, XOR_1)'
        data = TrialData(X, y, feature_labels, outlier_sets=[outlier_set])
        return data


class SingleOutlierQuadrantGenerator(XORGenerator):
    def __init__(self, noise_level=0.05, outlier_location=[0, 1]):
        super().__init__(noise_level)
        self.outlier_location = outlier_location

    def generate_inliers_outliers(self, n_inliers=1000, n_outliers=20, n_random_features=5):
        X_outliers, y_outliers = self._generate_data(self.outlier_location[0], self.outlier_location[1], n_outliers, label=-1)

        X_inliers_0, y_inliers_0 = self._generate_data(1 - self.outlier_location[0], 1 - self.outlier_location[1], n_inliers // 3, label=1) # TODO discuss: size // 3 or full size for each quadrant?
        X_inliers_1, y_inliers_1 = self._generate_data(self.outlier_location[0], 1 - self.outlier_location[1], n_inliers // 3, label=1)
        X_inliers_2, y_inliers_2 = self._generate_data(1 - self.outlier_location[0], self.outlier_location[1], n_inliers // 3, label=1)
        X_inliers = np.concatenate([X_inliers_0, X_inliers_1, X_inliers_2])
        y_inliers = np.concatenate([y_inliers_0, y_inliers_1, y_inliers_2])
        
        X = np.concatenate([X_inliers, X_outliers])
        y = np.concatenate([y_inliers, y_outliers])

        # add random noise features
        X = np.column_stack([X, self.noise_level*np.random.randn(X.shape[0], n_random_features)])

        feature_labels = ['XOR_0', 'XOR_1'] + [f'Rnd_{i}' for i in range(n_random_features)]
        # create tuples for each combination of features
        feature_labels += create_interaction_labels(feature_labels)

        outlier_set = OutlierSet("XOR_singlefeature_diagonal", idx=np.where(y == -1)[0], outlier_feature_labels=['XOR_0', 'XOR_1', '(XOR_0, XOR_1)'])
        data = TrialData(X, y, feature_labels, outlier_sets=[outlier_set])
        return data


