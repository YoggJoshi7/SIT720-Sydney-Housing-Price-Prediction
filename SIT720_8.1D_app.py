import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "sydney_house_price_model.joblib"

st.set_page_config(page_title="Sydney House Price Predictor", page_icon="🏠", layout="centered")

st.title("Sydney House Price Prediction")
st.caption("SIT720 8.1D — Machine Learning Mini Project")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Place sydney_house_price_model.joblib in the same folder as app.py.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
features = bundle["features"]

st.write("Enter the property characteristics used by the trained model.")

suburb = st.selectbox("Suburb", ["Chatswood", "Parramatta", "Blacktown"])
sale_method = st.selectbox("Sale Method", ["Auction", "Private Treaty", "Other / Unknown"])
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=2, step=1)
parking = st.number_input("Parking spaces", min_value=0, max_value=10, value=1, step=1)
land_size = st.number_input("Land size (m²)", min_value=0.0, max_value=5000.0, value=500.0, step=10.0)
sale_year = st.number_input("Sale year", min_value=2020, max_value=2030, value=2026, step=1)

if st.button("Predict Sale Price", type="primary"):
    row = pd.DataFrame([{
        "Suburb": suburb,
        "Sale Method": sale_method,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Parking": parking,
        "Land Size (m²)": land_size if land_size > 0 else np.nan,
        "Sale Year": sale_year
    }])

    # Keep only the features used by the saved pipeline.
    row = row[features]
    prediction = pipeline.predict(row)[0]

    st.success(f"Estimated sale price: A${prediction:,.0f}")

    st.info(
        "This is a machine-learning estimate based on the collected sample. "
        "It is not an official property valuation. The model does not capture "
        "all factors that can affect a property's market value."
    )
