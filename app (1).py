 
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_rf_model.pkl")

# Function to predict drop/retain
def predict_outcome(input_data):
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return "Drop" if prediction[0] == 1 else "Retain"

# Streamlit UI
st.title("Student Dropout Prediction")

st.write("Enter student details below to predict if they will **Drop** or **Retain**.")

# Input Form
with st.form("student_form"):
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    application_mode = st.selectbox("Application Mode", ["Online", "Offline"])
    course = st.number_input("Course Code", min_value=1, max_value=100, value=1)
    attendance = st.selectbox("Attendance Type", ["Daytime", "Evening"])
    prev_qualification = st.selectbox("Previous Qualification", ["High School", "Bachelor", "Master", "PhD"])
    nationality = st.selectbox("Nationality", ["Local", "International"])
    mother_qualification = st.selectbox("Mother's Qualification", ["High School", "Bachelor", "Master", "PhD"])
    father_qualification = st.selectbox("Father's Qualification", ["High School", "Bachelor", "Master", "PhD"])
    age_at_enrollment = st.number_input("Age at Enrollment", min_value=15, max_value=50, value=18)

    submit_button = st.form_submit_button("Predict Outcome")

# Convert categorical inputs to numerical
def encode_input():
    marital_status_dict = {"Single": 0, "Married": 1, "Divorced": 2}
    application_mode_dict = {"Online": 0, "Offline": 1}
    attendance_dict = {"Daytime": 0, "Evening": 1}
    qualification_dict = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    nationality_dict = {"Local": 0, "International": 1}

    return [
        marital_status_dict[marital_status],
        application_mode_dict[application_mode],
        course,
        attendance_dict[attendance],
        qualification_dict[prev_qualification],
        nationality_dict[nationality],
        qualification_dict[mother_qualification],
        qualification_dict[father_qualification],
        age_at_enrollment,
    ]

# Make prediction
if submit_button:
    input_data = encode_input()
    result = predict_outcome(input_data)
    st.success(f"🎯 Prediction: The student is likely to **{result}**")
