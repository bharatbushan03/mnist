# Handwritten Digit Recognizer ✍️

A complete end-to-end Machine Learning pipeline for handwritten digit recognition using the MNIST dataset. The project includes data fetching, model training (comparing Logistic Regression and Random Forest), a CLI prediction script, and an interactive Streamlit Web App.

## Features
- **Data Pipeline:** Automatically fetches the MNIST dataset via Scikit-Learn, normalizes pixel values, and splits it into training and testing sets.
- **Model Training:** Trains and evaluates both Logistic Regression and Random Forest models, automatically selecting and saving the best performer (`model.pkl`).
- **Command-Line Inference:** Includes a CLI tool (`prediction.py`) to process raw external images and predict the digit.
- **Streamlit UI:** A clean, user-friendly web interface (`app.py`) for uploading images and seeing real-time predictions.

## Project Structure
```text
mnist/
├── app.py                # Streamlit web application
├── main.py               # Basic script to load data and visualize samples
├── model_training.py     # Script to train, evaluate, and save the ML models
├── prediction.py         # CLI inference script to predict external images
├── requirements.txt      # Project dependencies
├── .gitignore            # Ignored files (including model.pkl)
└── src/
    ├── __init__.py
    ├── data_loader.py    # Functions for data loading and preprocessing
    └── visualization.py  # Plotting utilities for matplotlib
```

## Setup Instructions

### 1. Create a Virtual Environment
It is recommended to run this project inside a Python virtual environment.
```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

## Usage Guide

### Step 1: Train the Model
Because the serialized Random Forest model exceeds GitHub's file size limit, **you must train the model locally first** before running predictions.

Run the training pipeline:
```powershell
python model_training.py
```
*This will fetch the dataset, train the models, print evaluation metrics, generate confusion matrices, and save `model.pkl`.*

### Step 2: Test with the CLI (Optional)
You can use the prediction script to predict digits from image files via the command line.
```powershell
python prediction.py --image path/to/your/image.png
```

### Step 3: Launch the Streamlit Web App
To use the interactive UI, run the Streamlit app:
```powershell
streamlit run app.py
```
This will start a local web server and open the application in your default browser (usually at `http://localhost:8501`). You can then upload any `.png` or `.jpg` image containing a handwritten digit to get a prediction.

## Note on `.pkl` Files
The trained Random Forest model (`model.pkl`) is approximately 130MB, which exceeds GitHub's standard 100MB limit. As such, it is excluded via `.gitignore`. Please ensure you execute **Step 1** to generate the file locally on your machine.
