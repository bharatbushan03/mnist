"""
Main script for running the MNIST data preparation pipeline.
"""

from src.data_loader import load_mnist_data, preprocess_data, split_data
from src.visualization import plot_sample_digits

def main():
    """
    Main pipeline function to load, preprocess, and visualize the dataset.
    """
    print("=== MNIST Handwritten Digit Recognition Pipeline ===")
    
    # 1. Load the dataset
    X, y = load_mnist_data()
    
    # 2. Print initial dataset shape and sample labels
    print("\n--- Initial Dataset Info ---")
    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")
    print(f"Sample labels (first 10): {y[:10]}")
    
    # 3. Preprocess the dataset (Normalization)
    print("\n--- Preprocessing ---")
    X_preprocessed = preprocess_data(X)
    
    # 4. Split the dataset into training and testing sets
    print("\n--- Data Splitting ---")
    X_train, X_test, y_train, y_test = split_data(X_preprocessed, y, test_size=0.2, random_state=42)
    
    # 5. Print final shapes
    print("\n--- Final Dataset Shapes ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 6. Visualize a few digits from the training set
    # Note: We pass the preprocessed data, so pixel values are [0, 1]
    print("\n--- Visualization ---")
    plot_sample_digits(X_train, y_train, num_samples=5)
    
    print("\nData preparation complete. Ready for model training!")

if __name__ == "__main__":
    main()
