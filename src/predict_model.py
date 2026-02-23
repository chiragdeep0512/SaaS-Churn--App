# predict_model.py
import pandas as pd
import os
import joblib

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "processed", "new_customers.csv")
model_path = os.path.join(BASE_DIR, "models", "churn_model_xgb.pkl")

# -------------------------------
# Load new data
# -------------------------------
try:
    df_new = pd.read_csv(data_path)
    print("New data shape:", df_new.shape)
except Exception as e:
    print("Error reading CSV:", e)
    df_new = pd.DataFrame()

# -------------------------------
# Load trained model
# -------------------------------
try:
    model = joblib.load(model_path)
    print("Model loaded from:", model_path)
except Exception as e:
    print("Error loading model:", e)
    exit()

# -------------------------------
# Get original columns used in training
# -------------------------------
# This gets the original column names (before OneHotEncoding)
try:
    transformers = model.named_steps["preprocessor"].transformers_
    original_columns = []
    for name, trans, cols in transformers:
        original_columns.extend(cols)
except Exception as e:
    print("Error extracting original columns:", e)
    exit()

# -------------------------------
# Fill missing columns in new data
# -------------------------------
X_new = df_new.copy()

for col in original_columns:
    if col not in X_new.columns:
        # If the column was categorical in training
        if df_new.select_dtypes(include=["object"]).columns.isin([col]).any():
            X_new[col] = "missing"
        else:
            X_new[col] = 0

# Keep only the training columns
X_new = X_new[original_columns]

# -------------------------------
# Predict churn
# -------------------------------
try:
    y_prob = model.predict_proba(X_new)[:, 1]
    df_new["churn_prob"] = y_prob
    df_new["churn_flag"] = (y_prob >= 0.5).astype(int)
except Exception as e:
    print("Error during prediction:", e)
    exit()

# -------------------------------
# Insights
# -------------------------------
if "account_id" in df_new.columns:
    print("\n--- Churn Predictions ---")
    print(df_new[["account_id", "churn_prob", "churn_flag"]].head())
else:
    print(df_new[["churn_prob", "churn_flag"]].head())

# Top 5 high-risk
top5 = df_new.sort_values("churn_prob", ascending=False).head(5)
if "account_id" in df_new.columns:
    print("\nTop 5 high-risk customers:")
    print(top5[["account_id", "churn_prob", "churn_flag"]])
else:
    print("\nTop 5 high-risk customers (no account_id):")
    print(top5[["churn_prob", "churn_flag"]])

# -------------------------------
# Save predictions
# -------------------------------
output_path = os.path.join(BASE_DIR, "data", "processed", "new_customers_predictions.csv")
df_new.to_csv(output_path, index=False)
print(f"\nPredictions saved at: {output_path}")