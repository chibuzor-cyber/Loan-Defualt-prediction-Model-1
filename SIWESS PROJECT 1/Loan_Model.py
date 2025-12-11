# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 22:01:58 2025

@author: PRINCE CHIBUZOR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv('Loan_Dataset.csv')
#print(df)

#Cleaning Data set
#1) Checking missing value
x = data.isnull().sum()
#print(x)

#2) fill missing value
#3) Handle outliers
#4) Encode categorical Varaibles
labelencode = ['Loan_ID', 'Gender', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
encoder = LabelEncoder()

for label in labelencode:
    data[label] = encoder.fit_transform(data[label])
    
#Feature Scaling - Drop unnecessary columns first
X = data.drop(['Loan_Status', 'Loan_ID'], axis=1) 
y = data['Loan_Status']

#Scaling numeric features  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#training and splitting 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,  random_state=42)

#Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


#Prediction
new_applicant = pd.DataFrame([{
    'Age': 35,
    'Gender': 1,
    'Dependents': 0,
    'Education': 1,
    'Self_Employed': 0,
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1,
    'Property_Area': 2
}])
# Scale the new applicant data using the same scaler
new_applicant_scaled = scaler.transform(new_applicant)

#Prediction
prediction = model.predict(new_applicant_scaled)
print('New_Applicant:', prediction)


x = data["Age"]
y = data["LoanAmount"]

plt.figure(figsize=(10, 6))
plt.plot(x, y, color="blue")
plt.title("Loan Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


import joblib
joblib.dump(model, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved successfully!")
    
    
