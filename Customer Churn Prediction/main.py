import pandas as pd
from src.preprocessing import preprocess_data
from models.churn_model import ChurnModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("data/Churn_Modelling.csv")

# Preprocess data
X, y, scaler, le_gender, le_geo = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = ChurnModel()
model.train(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
