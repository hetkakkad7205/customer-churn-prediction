import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write("Explore customer churn data, review model performance, and predict whether a customer may churn.")

DATA_FILE = "cleaned_custumer_churn_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    required = {"Churn"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain a 'Churn' column.")
    return df

try:
    data = load_data()
except FileNotFoundError:
    st.error(f"Could not find `{DATA_FILE}`. Upload this CSV into the same GitHub repository folder as app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

def clean_target(value):
    value = str(value).strip().lower()
    if value in ["yes", "1", "true"]:
        return 1
    if value in ["no", "0", "false"]:
        return 0
    return np.nan

data["Churn"] = data["Churn"].apply(clean_target)
data = data.dropna(subset=["Churn"]).copy()
data["Churn"] = data["Churn"].astype(int)

if data.empty or data["Churn"].nunique() < 2:
    st.error("The Churn column must contain both Yes/1 and No/0 values.")
    st.stop()

X = data.drop(columns=["Churn", "CustomerID"], errors="ignore").copy()
y = data["Churn"]

if X.empty:
    st.error("No feature columns were found after removing Churn and CustomerID.")
    st.stop()

for col in X.columns:
    if X[col].dtype == "object":
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().sum() == X[col].notna().sum() and X[col].notna().any():
            X[col] = converted

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numeric_cols]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if numeric_cols:
    transformers.append(("numeric", numeric_pipeline, numeric_cols))
if categorical_cols:
    transformers.append(("categorical", categorical_pipeline, categorical_cols))
preprocessor = ColumnTransformer(transformers)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError as exc:
    st.error(f"Not enough examples in each churn class to create a stratified test split: {exc}")
    st.stop()

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, zero_division=0) * 100
recall = recall_score(y_test, y_pred, zero_division=0) * 100
f1 = f1_score(y_test, y_pred, zero_division=0) * 100
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

total_customers = len(data)
churn_count = int((y == 1).sum())
churn_rate = churn_count / total_customers * 100

dashboard, predict_tab, dataset_tab = st.tabs(["Dashboard & Evaluation", "Predict Churn", "Dataset"])

with dashboard:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Customers Churned", f"{churn_count:,}")
    c3.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.subheader("Model Evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.2f}%")
    m2.metric("Precision", f"{precision:.2f}%")
    m3.metric("Recall", f"{recall:.2f}%")
    m4.metric("F1 Score", f"{f1:.2f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Churn Distribution")
        counts = data["Churn"].map({0: "No Churn", 1: "Churn"}).value_counts()
        st.bar_chart(counts)
    with right:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, colorbar=False)
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Metrics are calculated on a held-out 20% test set. Model results depend on the dataset and selected features.")

with predict_tab:
    st.subheader("Enter Customer Details")
    st.write("Complete the fields below to estimate churn risk.")
    with st.form("churn_prediction_form"):
        row = {}
        columns = st.columns(2)
        for i, col in enumerate(X.columns):
            with columns[i % 2]:
                if col in numeric_cols:
                    median = X[col].median()
                    default = 0.0 if pd.isna(median) else float(median)
                    row[col] = st.number_input(
                        col.replace("_", " "),
                        value=default,
                        help=f"Numeric feature: {col}"
                    )
