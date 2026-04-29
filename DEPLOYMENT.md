# Deployment Guide: Handwritten Digit Recognizer

This guide will walk you through the process of deploying your Streamlit digit recognition application to the cloud.

---

## 📂 Final Project Structure
Before deploying, ensure your project structure looks exactly like this:

```text
mnist/
├── app.py                # Main Streamlit web application
├── prediction.py         # Prediction logic (imports model.pkl)
├── model.pkl             # Your trained Random Forest model
├── requirements.txt      # Required dependencies for the cloud server
├── main.py               # (Optional) Original pipeline script
├── model_training.py     # (Optional) Model training script
├── .gitignore            # Ignored files
└── src/                  # Source modules for training
    ├── __init__.py
    ├── data_loader.py
    └── visualization.py
```
*(Note: Ensure all file paths in `app.py` and `prediction.py` are relative. I have already verified this!)*

---

## 📦 `requirements.txt` Content
Your `requirements.txt` has been optimized and contains all necessary packages:
```text
scikit-learn
matplotlib
numpy
pandas
joblib
Pillow
streamlit
```

---

## 🚀 Deployment Option 1: Streamlit Community Cloud (Recommended)
Streamlit Community Cloud is the fastest way to deploy your app for free.

**Important Note on `model.pkl`:**
GitHub has a strict file size limit of 100MB. If your `model.pkl` is larger than this, you will need to either train a smaller model (e.g., Logistic Regression or a Random Forest with fewer estimators/shallower max depth) OR use [Git LFS](https://git-lfs.github.com/) to push the model file to your GitHub repository. The cloud server **must** be able to access `model.pkl` from your GitHub repository.

### Steps:
1. **Push to GitHub**: Commit your code and push it to a public or private GitHub repository.
2. **Log into Streamlit**: Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with your GitHub account.
3. **Deploy App**:
   - Click the **"New app"** button.
   - Select your repository, branch (e.g., `master`), and set the main file path to `app.py`.
   - Click **Deploy**.
4. **Auto-Deployment**: Streamlit will automatically install the packages listed in `requirements.txt` and launch your app. Any future `git push` to your repository will automatically update the live app.

---

## 🌐 Deployment Option 2: Render

Render is a versatile cloud platform that gives you more control over the deployment environment.

### Steps:
1. **Push to GitHub**: Push your repository to GitHub (ensure `model.pkl` is accessible).
2. **Log into Render**: Go to [https://render.com/](https://render.com/) and create an account.
3. **Create Web Service**:
   - Click **"New"** and select **"Web Service"**.
   - Connect your GitHub repository.
4. **Configure Service**:
   - **Name:** digit-recognizer (or similar)
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port=10000 --server.address=0.0.0.0`
5. **(Optional) Python Version**: You can specify the exact Python version by creating a file named `runtime.txt` in your project root containing just the version number (e.g., `python-3.11.0`).
6. **Deploy**: Click **"Create Web Service"**. Render will build the image, install dependencies, and start the app on the specified port.

---

## 🛠️ Common Errors & Fixes

1. **`FileNotFoundError: model.pkl not found`**
   - **Cause**: The model file wasn't pushed to GitHub because it was added to `.gitignore` or was too large.
   - **Fix**: Train a smaller model (reduce `n_estimators` in Random Forest) so it fits under 100MB, remove it from `.gitignore`, and push it to GitHub. Alternatively, use Git LFS.

2. **`ModuleNotFoundError: No module named 'X'`**
   - **Cause**: A required package is missing from `requirements.txt`.
   - **Fix**: Ensure `requirements.txt` is up-to-date and spelled correctly. (We have verified the current one is correct).

3. **App is very slow or crashes after uploading an image**
   - **Cause**: The model was being reloaded into memory on every single prediction. 
   - **Fix**: I have already optimized `app.py` by implementing `@st.cache_resource`. This ensures the model is loaded into memory only once when the server starts, making predictions blazing fast!

4. **Incompatible Python Version or Scikit-Learn Mismatch**
   - **Cause**: The scikit-learn version used to train the model differs from the cloud environment.
   - **Fix**: Freeze your scikit-learn version in `requirements.txt` (e.g., `scikit-learn==1.3.0`) and ensure it matches your local environment exactly.
