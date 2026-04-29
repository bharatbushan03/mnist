"""
Prediction pipeline for MNIST digit recognition.
Loads a trained model and predicts the digit from an external image file.
"""

import os
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def load_trained_model(model_path="model.pkl"):
    """
    Loads the trained machine learning model from disk.
    
    Args:
        model_path (str): Path to the serialized model file.
        
    Returns:
        model: The scikit-learn model object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please ensure the model is trained and saved.")
    
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model from '{model_path}'")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model. Ensure it is a valid pickle/joblib file. Error: {e}")

def preprocess_image(image_path):
    """
    Loads and preprocesses an image to match MNIST format.
    Steps: Grayscale -> Resize (28x28) -> Invert colors (if needed) -> Normalize (0-1) -> Flatten
    
    Args:
        image_path (str): Path to the input image file.
        
    Returns:
        img_flattened (np.ndarray): The 1D normalized array of size 784.
        img_array (np.ndarray): The 2D array of size 28x28 for visualization.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at '{image_path}'")
        
    # Supported formats check
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in supported_formats:
        raise ValueError(f"Unsupported image format: '{ext}'. Supported formats are: {supported_formats}")

    try:
        # Load image and convert to grayscale ('L' mode)
        img = Image.open(image_path).convert('L')
    except Exception as e:
        raise ValueError(f"Could not open or process the image file. It might be corrupted. Error: {e}")
        
    # Resize to 28x28 pixels using LANCZOS resampling (high quality)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert image to numpy array for numerical operations
    img_array = np.array(img)
    
    # MNIST digits are typically white (255) on a black (0) background.
    # Hand-drawn digits on paper are often dark on a light background.
    # If the image is mostly light (mean > 127), invert the colors.
    if np.mean(img_array) > 127:
        print("Light background detected. Inverting colors to match MNIST (white digit on black background)...")
        img = ImageOps.invert(img)
        img_array = np.array(img)
        
    # Normalize pixel values to [0.0, 1.0] to remain consistent with the training pipeline
    img_normalized = img_array / 255.0
    
    # Flatten the image to a 1D array of 784 pixels
    img_flattened = img_normalized.flatten()
    
    return img_flattened, img_array

def predict_digit(image_path: str, model_path: str = "model.pkl", display_plot: bool = True, model=None) -> int:
    """
    Predicts the digit from an image using the specified model.
    
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model.
        
    Returns:
        int: The predicted digit label.
    """
    # 1. Load the model
    if model is None:
        model = load_trained_model(model_path)
    
    # 2. Preprocess the external image
    print(f"Preprocessing image '{image_path}'...")
    img_flattened, img_original_shape = preprocess_image(image_path)
    
    # 3. Predict the digit
    # The model expects a 2D array of shape (n_samples, n_features)
    X_input = img_flattened.reshape(1, -1)
    
    prediction = model.predict(X_input)
    predicted_digit = int(prediction[0])
    
    # 4. Display the processed image that was fed into the model
    print(f"Displaying processed 28x28 image. Close the plot window to finish execution.")
    plt.figure(figsize=(4, 4))
    plt.imshow(img_original_shape, cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit}", color='green', fontsize=18, fontweight='bold')
    plt.axis('off')
    
    # Save a copy of the processed image in case it's run in a headless environment
    output_img_name = "processed_input_image.png"
    plt.savefig(output_img_name)
    if display_plot:
        plt.show(block=True)
    else:
        plt.close()
    
    return predicted_digit

def main():
    """
    Main function for handling CLI arguments and executing the prediction.
    """
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from an image using a trained model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file (e.g., test_image.png)")
    parser.add_argument("--model", type=str, default="model.pkl", help="Path to the trained model file (default: model.pkl)")
    
    args = parser.parse_args()
    
    try:
        predicted_digit = predict_digit(args.image, args.model)
        print("\n" + "="*40)
        print(f"  RESULT: The predicted digit is {predicted_digit}")
        print("="*40 + "\n")
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()
