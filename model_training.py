"""
Script for training, evaluating, and saving machine learning models for MNIST digit classification.
"""

import warnings
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
from sklearn.exceptions import ConvergenceWarning

from src.data_loader import load_mnist_data, preprocess_data, split_data


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model on the training data.
    """
    print("Training Logistic Regression model...")
    # Using 'saga' solver which is optimized for large datasets.
    # Suppress convergence warnings since MNIST might need more iterations 
    # to perfectly converge, but we cap it at 100 for speed.
    model = LogisticRegression(solver='saga', max_iter=100, random_state=42, n_jobs=-1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
        
    print("Logistic Regression training completed.")
    return model


def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest classifier on the training data.
    """
    print("Training Random Forest model (100 estimators)...")
    # n_jobs=-1 utilizes all available CPU cores for faster training
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Random Forest training completed.")
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a trained model, prints metrics, and returns accuracy.
    """
    print(f"\n--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, cm, y_pred


def plot_confusion_matrix(cm, model_name, labels):
    """
    Visualizes the confusion matrix.
    """
    print(f"Plotting confusion matrix for {model_name}...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    # Save the figure as well, so the user has an artifact if running without a display.
    plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.show()


def plot_predictions(X_test, y_test, y_pred, num_samples=6):
    """
    Visualizes test images with their predicted vs actual labels.
    """
    print(f"Visualizing {num_samples} test predictions...")
    plt.figure(figsize=(12, 5))
    
    # Randomly select samples from the test set
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image = X_test[idx].reshape(28, 28)
        actual_label = y_test[idx]
        predicted_label = y_pred[idx]
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image, cmap='gray')
        
        # Color text green for correct predictions, red for incorrect
        color = 'green' if actual_label == predicted_label else 'red'
        plt.title(f"Pred: {predicted_label}\nTrue: {actual_label}", color=color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.show()


def save_model(model, filepath="model.pkl"):
    """
    Saves the best-performing model to disk.
    """
    print(f"Saving the best model to {filepath}...")
    joblib.dump(model, filepath)
    print(f"Model saved successfully to {filepath}.")


def main():
    print("=== MNIST Model Training Pipeline ===")
    
    # 1. Load and preprocess data (reusing existing data loaders)
    X, y = load_mnist_data()
    X_preprocessed = preprocess_data(X)
    X_train, X_test, y_train, y_test = split_data(X_preprocessed, y, test_size=0.2, random_state=42)
    
    # 2. Train models
    print("\n--- Model Training ---")
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # 3. Evaluate models
    print("\n--- Model Evaluation ---")
    classes = np.unique(y)
    
    lr_acc, lr_cm, lr_pred = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_acc, rf_cm, rf_pred = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # 4. Compare performance
    print("\n" + "="*40)
    print("       MODEL COMPARISON SUMMARY")
    print("="*40)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Random Forest Accuracy:       {rf_acc:.4f}")
    print("="*40)
    
    if rf_acc > lr_acc:
        print("-> Random Forest performed better.")
        best_model = rf_model
        best_name = "Random Forest"
        best_pred = rf_pred
        best_cm = rf_cm
    else:
        print("-> Logistic Regression performed better.")
        best_model = lr_model
        best_name = "Logistic Regression"
        best_pred = lr_pred
        best_cm = lr_cm
        
    # 5. Visualize Confusion Matrix and Predictions for the best model
    print(f"\n--- Visualizations for {best_name} ---")
    plot_confusion_matrix(best_cm, best_name, classes)
    plot_predictions(X_test, y_test, best_pred, num_samples=6)
    
    # 6. Save the best model
    save_model(best_model, "model.pkl")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
