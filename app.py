import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('columns.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter the transaction details below to check if it's fraudulent.")

input_data = {}
for feature in feature_columns:
    if feature == 'Time':
        input_data[feature] = st.number_input(f'{feature}', value=1000.0)
    elif feature == 'Amount':
        input_data[feature] = st.number_input(f'{feature}', value=50.0)
    else:
        input_data[feature] = st.number_input(f'{feature}', value=0.0, step=0.01)

if st.button("Predict Fraud"):
    df = pd.DataFrame([input_data])
    df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    if prediction == 1:
        st.error(f"🚨 This transaction is **Fraudulent** with a probability of {probability:.2%}")
    else:
        st.success(f"✅ This transaction is **Legitimate** with a probability of {1 - probability:.2%}")
