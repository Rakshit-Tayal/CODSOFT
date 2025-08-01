import numpy as np
from src.preprocessing import preprocess_data
from models.churn_model import ChurnModel
import pandas as pd

# Load and preprocess dataset to reuse encoders and scalers
df = pd.read_csv("data/Churn_Modelling.csv")
X, y, scaler, le_gender, le_geo = preprocess_data(df)

# Retrain model
model = ChurnModel()
model.train(X, y)

# User input
print("Enter the following customer details:")
geo = input("Geography (France/Germany/Spain): ").capitalize()
gender = input("Gender (Male/Female): ").capitalize()
credit_score = int(input("Credit Score: "))
age = int(input("Age: "))
tenure = int(input("Tenure: "))
balance = float(input("Balance: "))
num_products = int(input("Number of Products: "))
has_cr_card = int(input("Has Credit Card? (1=yes, 0=no): "))
is_active = int(input("Is Active Member? (1=yes, 0=no): "))
salary = float(input("Estimated Salary: "))

# Format input in correct feature order
raw_input = pd.DataFrame([[
    credit_score,
    le_geo.transform([geo])[0],
    le_gender.transform([gender])[0],
    age,
    tenure,
    balance,
    num_products,
    has_cr_card,
    is_active,
    salary
]], columns=[
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
])

# Scale input
X_input = scaler.transform(raw_input)

# Predict
pred = model.predict(X_input)[0]
print("\nPrediction:", "Churn ❌" if pred == 1 else "No Churn ✅")
