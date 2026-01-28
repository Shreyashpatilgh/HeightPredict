import streamlit as st
import joblib
import numpy as np

st.title("Height Prediction App")

st.write("Enter your Age and Weight to predict Height:")

# User input fields
age = st.number_input("Age", min_value=0, max_value=120)
weight = st.number_input("Weight (kg)", min_value=0, max_value=200, value=70)

# Load model
model = joblib.load("height_predictor_model.pkl")

# Predict button
if st.button("Predict Height"):
    input_data = np.array([[age, weight]])
    predicted_height = model.predict(input_data)
    st.success(f"Predicted Height: {predicted_height[0]:.2f} cm")
