import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Waste Type Classifier",
    page_icon="♻️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_model.h5")

model = load_model()

# Class names (must match training)
class_names = ["glass", "metal", "paper", "plastic"]

# Colors for each waste type
color_map = {
    "plastic": "#FB8C00",
    "paper": "#1E88E5",
    "glass": "#43A047",
    "metal": "#546E7A"
}

# ---------------- UI ----------------
st.markdown("""
<h1 style="text-align:center; color:#1B5E20;">♻️ Smart Waste Type Classifier</h1>
<p style="text-align:center;">
Upload one or more waste images and classify them using a CNN-based ML model
</p>
<hr>
""", unsafe_allow_html=True)

# Multiple image upload
uploaded_files = st.file_uploader(
    "📤 Upload waste images (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.markdown("### 🔍 Uploaded Images")

    if st.button("🚀 CLASSIFY ALL IMAGES"):
        for uploaded_file in uploaded_files:

            # Load and show image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, width=300)

            # Preprocess image (same as training)
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            preds = model.predict(img_array)
            confidence = float(np.max(preds))
            predicted_class = class_names[np.argmax(preds)]

            # Display result
            st.markdown(
                f"""
                <div style="
                    background:{color_map[predicted_class]};
                    padding:18px;
                    border-radius:16px;
                    color:white;
                    text-align:center;
                    margin-top:10px;">
                    <h3>🗑️ Waste Type: {predicted_class.upper()}</h3>
                    <p>🎯 Confidence: {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence)
