import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
model = joblib.load("models/sleep_quality_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define the categories
occupation_categories = ["Student", "Employee", "Self-employed", "Unemployed"]
bmi_categories = ["Underweight", "Normal", "Overweight", "Obese"]
sleep_disorder_categories = ["None", "Insomnia", "Sleep Apnea", "Other"]

# Create and fit label encoders
occupation_le = LabelEncoder()
bmi_le = LabelEncoder()
sleep_disorder_le = LabelEncoder()

occupation_le.fit(occupation_categories)
bmi_le.fit(bmi_categories)
sleep_disorder_le.fit(sleep_disorder_categories)

# Streamlit App UI
st.title("🛌 Sleep Quality Prediction App")
st.write("Enter your details to check if your sleep quality is **Good** or **Bad**.")

# User Inputs
age = st.slider("Age", 18, 90, 25)
occupation = st.selectbox("Occupation", occupation_categories)
sleep_duration = st.slider("Sleep Duration (hours)", 3, 12, 7)
physical_activity = st.slider("Physical Activity Level (0-10)", 0, 10, 5)
stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
bmi_category = st.selectbox("BMI Category", bmi_categories)
heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=120, value=72)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)
sleep_disorder = st.selectbox("Sleep Disorder", sleep_disorder_categories)

# Convert categorical values using respective LabelEncoders
occupation_encoded = occupation_le.transform([occupation])[0]
bmi_encoded = bmi_le.transform([bmi_category])[0]
sleep_disorder_encoded = sleep_disorder_le.transform([sleep_disorder])[0]

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
