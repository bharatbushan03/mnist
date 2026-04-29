"""
Module for data loading and preprocessing.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

def load_mnist_data():
    """
    Loads the MNIST dataset using scikit-learn's fetch_openml.
    
    Returns:
        X (np.ndarray): The feature matrix (images).
        y (np.ndarray): The target labels.
    """
    print("Loading MNIST dataset... This might take a minute.")
    # as_frame=False ensures we get numpy arrays instead of pandas DataFrames
    # cache=True prevents downloading the data multiple times
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    
    # Convert labels to integers (they are downloaded as strings by default)
    y = y.astype(np.uint8)
    
    print(f"Dataset successfully loaded! Total samples: {X.shape[0]}, Features: {X.shape[1]}")
    return X, y

def preprocess_data(X):
    """
    Preprocesses the features.
    For MNIST, it normalizes pixel values from [0, 255] to [0.0, 1.0].
    
    Args:
        X (np.ndarray): The feature matrix.
        
    Returns:
        X_normalized (np.ndarray): The normalized feature matrix.
    """
    print("Preprocessing data: Normalizing pixel values to [0, 1].")
    X_normalized = X / 255.0
    return X_normalized

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test: The split datasets.
    """
    print(f"Splitting data into training ({(1-test_size)*100}%) and testing ({test_size*100}%) sets.")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
