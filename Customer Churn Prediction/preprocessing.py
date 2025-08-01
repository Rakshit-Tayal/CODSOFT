import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    # Drop unused columns
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    # Encode categorical variables
    le_gender = LabelEncoder()
    le_geo = LabelEncoder()
    df["Gender"] = le_gender.fit_transform(df["Gender"])
    df["Geography"] = le_geo.fit_transform(df["Geography"])

    # Separate features and target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le_gender, le_geo
