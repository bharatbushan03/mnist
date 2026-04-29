"""
Module for data visualization.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_sample_digits(X, y, num_samples=5):
    """
    Visualizes a few sample digits from the dataset.
    
    Args:
        X (np.ndarray): The feature matrix (flattened images).
        y (np.ndarray): The labels.
        num_samples (int): Number of samples to visualize.
    """
    print(f"Visualizing {num_samples} sample digits...")
    plt.figure(figsize=(10, 4))
    
    # Randomly select samples to ensure variety
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # The MNIST dataset features are flattened 784 arrays
        # We need to reshape them back to 28x28 images for visualization
        image = X[idx].reshape(28, 28)
        label = y[idx]
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
