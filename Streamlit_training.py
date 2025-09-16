import streamlit as st
st.set_page_config(page_title="Realtime Machine Recognition", layout="wide")

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing.image import img_to_array


# ---------------------------
# Setup
# ---------------------------
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained MobileNetV2 model
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

st.title("🤖 Machine Recognition System")

# ---------------------------
# Prediction Function
# ---------------------------
def predict_image(img: Image.Image):
    """Preprocess and predict image class."""
    img_resized = img.resize((224, 224))
    x = img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded

# ---------------------------
# Session State to store results
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Tabs for Upload vs Camera
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📤 Upload Images", "📸 Camera Scan", "📊 Results"])

with tab1:
    st.subheader("Upload Machine Images")
    uploaded_files = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        cols = st.columns(3)
        for i, uploaded_file in enumerate(uploaded_files):
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            img = Image.open(file_path).convert("RGB")
            decoded = predict_image(img)
            label, conf = decoded[0][1], float(decoded[0][2]) * 100

            # Save to history
            st.session_state.history.append({
                "Filename": uploaded_file.name,
                "Predicted Label": label,
                "Confidence (%)": round(conf, 2),
                "Scan Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Source": "Uploaded"
            })

            with cols[i % 3]:
                st.image(
                    img,
                    caption=f"{label} ({conf:.2f}%)",
                    use_container_width=True
                )

with tab2:
    st.subheader("Realtime Camera Scan")
    camera_image = st.camera_input("Take a picture of a machine")

    if camera_image:
        img = Image.open(camera_image).convert("RGB")
        decoded = predict_image(img)
        label, conf = decoded[0][1], float(decoded[0][2]) * 100

        # Save to history
        st.session_state.history.append({
            "Filename": "camera_snapshot.png",
            "Predicted Label": label,
            "Confidence (%)": round(conf, 2),
            "Scan Time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Source": "Camera"
        })

        st.image(img, caption=f"Prediction: {label} ({conf:.2f}%)", use_container_width=True)
        st.success(f"✅ Machine Detected: **{label}** with {conf:.2f}% confidence")

with tab3:
    st.subheader("Prediction Results History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        # Download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv,
            file_name="machine_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("No scans yet. Upload an image or use the camera to start.")
