import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Random Forest Regression App",
    page_icon="🌲",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

# App title
st.title("🌲 Random Forest Regression Predictor")
st.write("Enter feature values to get prediction")

# ---- INPUT SECTION ----
st.subheader("🔢 Input Features")

# ⬇️ CHANGE THESE FEATURE NAMES BASED ON YOUR MODEL ⬇️
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)
feature_4 = st.number_input("Feature 4", value=0.0)

# Predict button
if st.button("📊 Predict"):
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])
    prediction = model.predict(input_data)[0]

    st.success(f"✅ Predicted Value: **{prediction:.2f}**")
