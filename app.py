import streamlit as st
import numpy as np
import joblib

# Load the trained model, scaler, and encoder
model = joblib.load("models/sleep_quality_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

# Streamlit App UI
st.title("🛌 Sleep Quality Prediction App")
st.write("Enter your details to check if your sleep quality is **Good** or **Bad**.")

# User Inputs
age = st.slider("Age", 18, 90, 25)
occupation = st.selectbox("Occupation", ["Student", "Employee", "Self-employed", "Unemployed"])
sleep_duration = st.slider("Sleep Duration (hours)", 3, 12, 7)
physical_activity = st.slider("Physical Activity Level (0-10)", 0, 10, 5)
stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=120, value=72)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)
sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea", "Other"])

# Convert categorical values using LabelEncoder
occupation_encoded = le.transform([Occupation])[0]
bmi_encoded = le.transform([BMI_Category])[0]
sleep_disorder_encoded = le.transform([Sleep_Disorder])[0]

# Scale numeric features
scaled_features = scaler.transform([[age, physical_activity, daily_steps]])

# Prepare input for model
input_data = np.array([[
    scaled_features[0][0], occupation_encoded, sleep_duration, 
    scaled_features[0][1], stress_level, bmi_encoded, heart_rate, 
    scaled_features[0][2], sleep_disorder_encoded
]])

# Prediction Button
if st.button("Predict Sleep Quality"):
    prediction = model.predict(input_data)[0]
    result = "Good 😊" if prediction == 1 else "Bad 😞"
    st.success(f"Your Sleep Quality is: **{result}**")
