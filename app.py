import streamlit as st
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("churn_model.keras")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer will stay or leave using machine learning.")

# User inputs
age = st.number_input("Age", 18, 100)
gender_label = st.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender_label == "Male" else 0
tenure = st.number_input("Tenure (months)", 0, 100)
usage = st.number_input("Usage Frequency", 0.0, 10.0)
support_calls = st.number_input("Customer Support Calls", 0, 10)
contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
contract_encoded = ["Month-to-Month", "One Year", "Two Year"].index(contract_type)

# Add rest of required numeric/encoded features here (as per training order)
# Example dummy values for illustration
features = np.array([[age, gender, tenure, usage, support_calls, contract_encoded, 450.5, 8, 0, 1, 0, 0, 1, 0]])

# Prediction button
if st.button("🔍 Predict"):
    pred = model.predict(features)
    st.write("### Probability:", round(float(pred[0][0]), 2))
    st.success("✅ Prediction: **Churn**" if pred > 0.5 else "🟩 Prediction: **Stay**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Created by <b>Ayesha Owais</b></p>", unsafe_allow_html=True)
