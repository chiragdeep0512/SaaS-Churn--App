import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1️⃣ Load dataset
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")

df = pd.read_csv(file_path)

X = df.drop(columns=["churned", "account_id", "account_name"])
y = df["churned"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------
# 2️⃣ Identify column types
# -------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# -------------------------
# 3️⃣ Preprocessing pipelines
# -------------------------
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_pipeline, categorical_cols),
    ("num", numerical_pipeline, numerical_cols)
])

# -------------------------
# 4️⃣ Build XGBoost pipeline
# -------------------------
# scale_pos_weight = ratio of negative / positive class to handle imbalance
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ))
])

# -------------------------
# 5️⃣ Train model
# -------------------------
model.fit(X_train, y_train)

# -------------------------
# 6️⃣ Predictions
# -------------------------
y_prob = model.predict_proba(X_test)[:, 1]

# Threshold tuning for better recall/precision
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

y_pred = (y_prob >= best_threshold).astype(int)

# -------------------------
# 7️⃣ Evaluation
# -------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)
pr_auc = auc(recall, precision)
print(f"ROC-AUC Score: {roc:.3f}")
print(f"PR-AUC Score: {pr_auc:.3f}")
print("Best threshold used:", round(best_threshold, 2))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()

print("\nUnique predictions:", pd.Series(y_pred).value_counts())
print("Average predicted probability:", y_prob.mean())

# -------------------------
# 8️⃣ Feature importance
# -------------------------
# Extract feature names after preprocessing
preprocessed_features = model.named_steps["preprocessor"].get_feature_names_out()
importances = model.named_steps["classifier"].feature_importances_
feat_imp_df = pd.Series(importances, index=preprocessed_features).sort_values(ascending=False)

print("\nTop 10 Feature Importances:")
print(feat_imp_df.head(10))

# Optional: plot
plt.figure(figsize=(10,6))
feat_imp_df.head(10).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.show()

# -------------------------
# 9️⃣ Save model
# -------------------------
model_path = os.path.join(BASE_DIR, "models", "churn_model_xgb.pkl")
joblib.dump(model, model_path)
print(f"\nModel saved at: {model_path}")