import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your pre-trained model
model = load_model(r'C:\Users\sreec\OneDrive\Desktop\DF\deepfake_detector_model.h5')

def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_deepfake(img):
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)[0][0]
    if prediction > 0.5:
        return 'FAKE'
    else:
        return 'REAL'

# Streamlit App Interface
st.title('Deepfake Image Detector')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Predict if the image is real or fake
    prediction = predict_deepfake(img)
    st.write(f"Prediction: {prediction}")
