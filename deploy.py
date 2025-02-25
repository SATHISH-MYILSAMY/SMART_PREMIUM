import mlflow
import xgboost as xgb
import streamlit as st
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("train.csv")

X = train_data[['Age', 'Annual Income', 'Health Score']].dropna()
y = train_data['Premium Amount'].loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

with mlflow.start_run() as run:
    mlflow.xgboost.log_model(xgb_model, "model", signature=infer_signature(X_train, y_train))

st.title("Insurance Premium Prediction")
age = st.number_input("Age", min_value=18, max_value=100, step=1)
income = st.number_input("Annual Income (in INR)", min_value=1000, step=100)
health_score = st.number_input("Health Score", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict"):
    input_data = pd.DataFrame([[age, income, health_score]], columns=['Age', 'Annual Income', 'Health Score'])
    prediction = xgb_model.predict(input_data)
    st.success(f"Predicted Insurance Premium: ₹{prediction[0]:.2f}")
