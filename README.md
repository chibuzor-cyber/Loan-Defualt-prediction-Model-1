# Loan-Defualt-prediction-Model-1
The Loan Default Prediction Model is a machine-learning system designed to predict whether an applicant is likely to default on a loan (or whether a loan should be approved or not). Using historical loan application data, the model learns patterns that distinguish approved applicants from those who are likely to default.
This project focuses on predicting whether a loan applicant is likely to
**default** or **repay** a loan using machine learning techniques. The
model is built using a structured dataset that contains customer
demographic information, financial history, and loan-related variables.

## 🎯 Objectives of the Project

-   Data Cleaning\
-   Exploratory Data Analysis (EDA)\
-   Feature Encoding\
-   Feature Scaling\
-   Model Training\
-   Model Evaluation\
-   Prediction\
-   Deployment with Streamlit

## 📂 Project Structure

    Loan-Default-Prediction/
    │
    ├── data/
    │   └── loan_data.csv
    │
    ├── notebooks/
    │   └── EDA.ipynb
    │
    ├── src/
    │   ├── preprocessing.py
    │   ├── model_training.py
    │   └── prediction.py
    │
    ├── model/
    │   └── loan_model.pkl
    │
    ├── app/
    │   └── streamlit_app.py
    │
    └── README.md

## 📊 Dataset Description

  Feature             Description
  ------------------- -------------------------
  Gender              Male/Female
  Married             Yes/No
  Dependents          Number of dependents
  Education           Graduate / Not Graduate
  Self_Employed       Yes/No
  ApplicantIncome     Income of applicant
  CoapplicantIncome   Income of co-applicant
  LoanAmount          Loan amount requested
  Loan_Amount_Term    Loan repayment duration
  Credit_History      1 = Good, 0 = Bad
  Property_Area       Urban/Semiurban/Rural
  Loan_Status         Target variable

## 🧹 Data Preprocessing

-   Handling missing values\
-   Label Encoding & OneHot Encoding\
-   Standard Scaling\
-   Train-test split

## 🤖 Model Development

Models: - Logistic Regression\
- Decision Tree\
- Random Forest\
- SVM

## 📈 Evaluation Metrics

-   Accuracy\
-   Precision\
-   Recall\
-   F1 Score\
-   Confusion Matrix

## 🔮 Making Predictions

Example input:

    {
     "Gender": "Male",
     "Married": "Yes",
     "Dependents": "2",
     "Education": "Graduate",
     "Self_Employed": "No",
     "ApplicantIncome": 4500,
     "CoapplicantIncome": 1500,
     "LoanAmount": 120,
     "Loan_Amount_Term": 360,
     "Credit_History": 1,
     "Property_Area": "Urban"
    }

## 🌐 Deployment

Run the Streamlit app:

    streamlit run app/streamlit_app.py

## 📝 Conclusion

The model helps financial institutions reduce risk and make smarter loan
decisions while demonstrating machine learning and data analytics
skills.
