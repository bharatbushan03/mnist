"""
Streamlit Web App for Handwritten Digit Recognition.
"""

import os
import streamlit as st
import tempfile
from PIL import Image
from prediction import predict_digit

import joblib

# Cache the model so it loads only once when the app starts
@st.cache_resource
def get_model(model_path="model.pkl"):
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please upload it or train the model locally.")
        st.stop()
    return joblib.load(model_path)

# Load the cached model
trained_model = get_model()

# Set up page config
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="centered"
)

# Sidebar with Project Info
with st.sidebar:
    st.title("✍️ Digit Recognizer")
    st.markdown("""
    ### About this App
    This web app predicts handwritten digits (0-9) from uploaded images.
    
    **Pipeline:**
    1. Upload an image
    2. Image is preprocessed (grayscale, 28x28 resize, color inversion, normalization)
    3. Random Forest model runs inference
    4. Prediction is displayed
    
    **Supported Formats:** JPG, JPEG, PNG
    """)
    st.info("Built with Scikit-Learn and Streamlit.")

st.title("Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit, and the model will predict what number it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Check format implicitly by trying to open
    try:
        img = Image.open(uploaded_file)
        
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Uploaded Image")
            # width='stretch' is the modern replacement for use_container_width=True
            st.image(img, width="stretch")
            
        if st.button("Predict Digit", type="primary"):
            with st.spinner("Processing image and predicting..."):
                try:
                    # Save the uploaded file temporarily so prediction.py can read it from path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                        temp_path = temp_file.name
                        img.save(temp_path)
                    
                    # Call predict_digit
                    # We pass display_plot=False to avoid blocking the Streamlit server
                    # We pass model=trained_model to utilize Streamlit's caching
                    predicted_digit = predict_digit(temp_path, model=trained_model, display_plot=False)
                    
                    # Display Prediction
                    st.success("Prediction complete!")
                    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>Predicted Digit: {predicted_digit}</h1>", unsafe_allow_html=True)
                    
                    # Display processed image (saved by prediction.py)
                    with col2:
                        st.subheader("Processed 28x28 Image")
                        if os.path.exists("processed_input_image.png"):
                            processed_img = Image.open("processed_input_image.png")
                            st.image(processed_img, width="stretch", caption="What the model sees")
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
    except Exception as e:
        st.error(f"Invalid image format or corrupted file. Error: {e}")
else:
    st.info("Please upload an image file to get started.")
