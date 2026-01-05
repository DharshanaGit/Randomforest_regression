import streamlit as st
import pandas as pd
import joblib
import openpyxl

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🏡 House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

# ---------------- LOAD MODEL & DATA ----------------
@st.cache_resource
def load_model():
    return joblib.load("rf_regression.pkl")

@st.cache_data
def load_data():
    return pd.read_excel("realistic_housing_data.xlsx")

model = load_model()
df = load_data()

# ---------------- UI ----------------
st.title("🏡 House Price Prediction App")
st.write("Enter house details to predict the **price**")

st.subheader("📊 House Details")

# Numerical inputs
bedrooms = st.number_input("Bedrooms", min_value=0, value=2)
bathrooms = st.number_input("Bathrooms", min_value=0, value=2)
sqft = st.number_input("Square Feet", min_value=100, value=1200)
lot_size = st.number_input("Lot Size (sqft)", min_value=100, value=2000)
age = st.number_input("House Age (years)", min_value=0, value=10)
year_built = st.number_input("Year Built", min_value=1900, value=2015)

# Categorical inputs (taken from training data)
garage = st.selectbox("Garage", df["garage"].unique())
location = st.selectbox("Location", df["location"].unique())
house_type = st.selectbox("House Type", df["house_type"].unique())
condition = st.selectbox("Condition", df["condition"].unique())

has_pool = st.selectbox("Has Pool", df["has_pool"].unique())
has_fireplace = st.selectbox("Has Fireplace", df["has_fireplace"].unique())
has_basement = st.selectbox("Has Basement", df["has_basement"].unique())

school_rating = st.slider("School Rating", 1, 10, 5)

# ---------------- PREDICTION ----------------
if st.button("💰 Predict Price"):
    # Create DataFrame with EXACT column names used in training
    input_df = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "lot_size": lot_size,
        "age": age,
        "year_built": year_built,
        "garage": garage,
        "location": location,
        "house_type": house_type,
        "condition": condition,
        "has_pool": has_pool,
        "has_fireplace": has_fireplace,
        "has_basement": has_basement,
        "school_rating": school_rating
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"🏷️ Estimated House Price: **₹ {prediction:,.2f}**")
