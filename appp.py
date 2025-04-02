
import streamlit as st
import joblib
import numpy as np
import pandas as pd
# Load the trained model
model = joblib.load("best_rf_model.pkl")

# Function to preprocess and reshape input
def encode_input():
    # Collect input from the user (Ensure exactly 34 inputs)
    user_inputs = [
        st.number_input(f"Feature {i+1}", value=0.0) for i in range(34)
    ]

    # Convert to NumPy array and reshape it to (1, 34)
    input_array = np.array(user_inputs).reshape(1, -1)
    return input_array

# Function to predict dropout/retention
def predict_outcome(input_data):
    prediction = model.predict(input_data)  # Now input_data has correct shape (1, 34)
    return "Drop" if prediction[0] == 1 else "Retain"

# Streamlit UI
st.title("Student Dropout Prediction")

# Collect user input
st.header("Enter Student Data")
input_data = encode_input()

# Predict button
if st.button("Predict"):
    result = predict_outcome(input_data)
    st.success(f"🎯 Prediction: The student is likely to **{result}**")

# Debugging information (Remove this after testing)
st.write(f"Shape of input_data: {input_data.shape}")  # Should be (1, 34) 

