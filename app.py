# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "churn_model_xgb.pkl")

# Load model
model = joblib.load(model_path)

st.title("🚀 Customer Churn Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload new customers CSV", type=["csv"])
if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    st.write("New data preview:", df_new.head())

    # -------------------------------
    # Prepare input (same fix as predict_model.py)
    # -------------------------------
    transformers = model.named_steps["preprocessor"].transformers_
    original_columns = []
    for name, trans, cols in transformers:
        original_columns.extend(cols)

    X_new = df_new.copy()
    for col in original_columns:
        if col not in X_new.columns:
            if df_new.select_dtypes(include=["object"]).columns.isin([col]).any():
                X_new[col] = "missing"
            else:
                X_new[col] = 0
    X_new = X_new[original_columns]

    # Predict
    y_prob = model.predict_proba(X_new)[:, 1]
    df_new["churn_prob"] = y_prob
    df_new["churn_flag"] = (y_prob >= 0.5).astype(int)

    st.subheader("Predictions")
    st.write(df_new[["churn_prob", "churn_flag"]].head())

    st.subheader("Top 5 High-Risk Customers")
    top5 = df_new.sort_values("churn_prob", ascending=False).head(5)
    st.write(top5[["churn_prob", "churn_flag"]])

    # Download CSV
    csv = df_new.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="new_customers_predictions.csv",
        mime="text/csv"
    )