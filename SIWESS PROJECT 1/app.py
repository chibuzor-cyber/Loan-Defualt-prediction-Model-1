import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

#Load model and scaler
model = joblib.load("Loan_Model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Default Prediction")
st.write("Enter the Applicant details to predict laon Approval")


#User input Form
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Graduate", "Not Graduate",])
self_Employ = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurbal"])

applicant_income = st.number_input("Applicant Income", min_value=0)
co_applicant_income = st.number_input("coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term", min_value=0)
credit_history = st.selectbox("Credit History", ["Good", "Bad"])
dependents = st.number_input("Dependents", min_value=0)
age = st.number_input("Age", min_value=18, max_value=100)

#Converting numeric to string 
if credit_history == "Good":
    credit_history = 1
else:
    credit_history = 0

# --- ENCODING MATCHING TRAINING ---
gender_val = 1 if gender == "Male" else 0
education_val = 1 if education == "Graduate" else 0
self_emp_val = 1 if self_Employ == "Yes" else 0

area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_val = area_map[Property_Area]

# Create input DataFrame with user inputs (matching training data structure)
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender_val,
    'Dependents': dependents,
    'Education': education_val,
    'Self_Employed': self_emp_val,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': co_applicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,
    'Property_Area': property_val
}])

# Scale ALL numeric values (must match training features)
# Use transform on the entire dataset with correct column order
input_data_scaled = scaler.transform(input_data)

# --- PREDICT ---
if st.button("Predict Loan Approval"):
    pred = model.predict(input_data_scaled)[0]

    if pred == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")

# --- SIMPLE PLOT ---
st.subheader("📊 Visual Analysis (Age vs Loan Amount)")

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(input_data["Age"], input_data["LoanAmount"], s=100)
ax.set_xlabel("Age")
ax.set_ylabel("Scaled Loan Amount")
ax.set_title("Applicant Loan Visualization")
st.pyplot(fig)