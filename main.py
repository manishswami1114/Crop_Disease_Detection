import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import json


def upload_image_case_insensitive(label="Upload an image"):
    extensions = ["jpg", "jpeg", "png", "bmp", "webp", "tiff"]
    all_exts = extensions + [ext.upper() for ext in extensions]
    return st.file_uploader(label, type=all_exts)


# Load your trained model
cnn = tf.keras.models.load_model('./trained_plant_disease_model.keras')

# Load class names (replace with actual path if different)
with open('./class_name.json', 'r') as f:
    class_name = json.load(f)

# Streamlit UI
st.title("Crop Disease Prediction using OpenCV")
st.write("Upload an image of the crop leaf to predict the disease.")

# File uploader
uploaded_file = upload_image_case_insensitive("Upload a crop image")

if uploaded_file is not None:
    # Convert file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)  
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  

    
    img_resized = cv2.resize(img_cv, (128, 128))
    input_arr = np.expand_dims(img_resized, axis=0)  

    
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)
    model_prediction = class_name[result_index]

    
    st.image(img_cv, caption='Uploaded Image (Original)', use_column_width=True)

    
    st.write(f"🩺 **Predicted Disease:** `{model_prediction}`")

    
    fig, ax = plt.subplots()
    ax.imshow(img_resized)
    ax.set_title(f"Disease Name: {model_prediction}")
    ax.axis('off')
    st.pyplot(fig)
