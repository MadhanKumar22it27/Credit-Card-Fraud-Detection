# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import joblib

# # Load model, scaler, and columns
# model = joblib.load('fraud_model.pkl')
# scaler = joblib.load('scaler.pkl')
# feature_columns = joblib.load('columns.pkl')

# app = FastAPI(title="Credit Card Fraud Detection API")

# class Transaction(BaseModel):
#     time: float
#     amount: float
#     V1: float
#     V2: float
#     V3: float
#     V4: float
#     V5: float
#     V6: float
#     V7: float
#     V8: float
#     V9: float
#     V10: float
#     V11: float
#     V12: float
#     V13: float
#     V14: float
#     V15: float
#     V16: float
#     V17: float
#     V18: float
#     V19: float
#     V20: float
#     V21: float
#     V22: float
#     V23: float
#     V24: float
#     V25: float
#     V26: float
#     V27: float
#     V28: float

# @app.post('/predict')
# def predict(transaction: Transaction):
#     # Convert input to DataFrame
#     input_data = transaction.dict()
#     df = pd.DataFrame([input_data])

#     # Rename 'time' and 'amount' to match training
#     df.rename(columns={'time': 'Time', 'amount': 'Amount'}, inplace=True)

#     # Add missing columns if any
#     for col in feature_columns:
#         if col not in df.columns:
#             df[col] = 0  # Default fill if column missing

#     # Ensure the column order is exactly like training
#     df = df[feature_columns]

#     # Scale 'Amount' and 'Time'
#     df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])

#     # Prediction
#     prediction = model.predict(df)[0]
#     probability = model.predict_proba(df)[0][1]

#     return {
#         'is_fraud': bool(prediction),
#         'fraud_probability': round(probability, 4)
#     }

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model, scaler, and columns
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('columns.pkl')

app = FastAPI(title="Credit Card Fraud Detection API")

# Request body schema
class Transaction(BaseModel):
    time: float
    amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()
    df = pd.DataFrame([data])

    df.rename(columns={'time': 'Time', 'amount': 'Amount'}, inplace=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        'is_fraud': bool(prediction),
        'fraud_probability': round(probability, 4)
    }