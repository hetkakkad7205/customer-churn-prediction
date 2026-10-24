
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

app = Flask(__name__)

# Load your already-cleaned and feature-engineered dataset
data = pd.read_csv("cleaned_custumer_churn_data.csv")

# Clean target labels
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

# Separate features and target
X = data.drop(
    columns=["Churn", "CustomerID"],
    errors="ignore"
).copy()

y = data["Churn"]

# Convert numerical-looking columns to numbers where appropriate
for col in X.columns:
    if X[col].dtype == "object":
        converted = pd.to_numeric(X[col], errors="coerce")

        if converted.notna().sum() == X[col].notna().sum():
            X[col] = converted

# Identify column types
numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = [
    col for col in X.columns if col not in numeric_cols
]

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# Split data for evaluation
if y.nunique() < 2:
    raise ValueError("Churn must contain both 0 and 1 classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build and train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# Evaluate on test data
y_pred = model.predict(X_test)

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
    "precision": round(
        precision_score(y_test, y_pred, zero_division=0) * 100, 2
    ),
    "recall": round(
        recall_score(y_test, y_pred, zero_division=0) * 100, 2
    ),
    "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
}

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Dashboard information
total_customers = len(data)
churn_count = int((y == 1).sum())
churn_rate = round(churn_count / total_customers * 100, 2)

# Generate form fields from the actual dataset
form_fields = []

for col in X.columns:
    if col in numeric_cols:
        median = X[col].median()
        default = 0 if pd.isna(median) else float(median)

        form_fields.append({
            "name": col,
            "label": col.replace("_", " "),
            "type": "number",
            "default": default,
            "options": []
        })
    else:
        options = sorted(
            X[col].dropna().astype(str).unique().tolist()
        )

        form_fields.append({
            "name": col,
            "label": col.replace("_", " "),
            "type": "select",
            "default": options[0] if options else "",
            "options": options
        })


@app.route("/")
def index():
    return render_template(
        "index.html",
        total=total_customers,
        churn_count=churn_count,
        churn_rate=churn_rate,
        metrics=metrics,
        matrix=cm.tolist()
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        row = {}

        try:
            for col in X.columns:
                value = request.form.get(col, "").strip()

                if col in numeric_cols:
                    if not value:
                        raise ValueError(
                            f"Please enter a value for {col}."
                        )
                    row[col] = float(value)
                else:
                    row[col] = value

            input_data = pd.DataFrame([row], columns=X.columns)

            prediction = int(model.predict(input_data)[0])
            result = "Yes" if prediction == 1 else "No"

            # Probability estimated by the model
            probabilities = model.predict_proba(input_data)[0]
            class_index = list(
                model.named_steps["classifier"].classes_
            ).index(1)

            probability = round(
                float(probabilities[class_index]) * 100, 2
            )

        except (ValueError, TypeError) as exc:
            error = str(exc)

    return render_template(
        "predict.html",
        fields=form_fields,
        result=result,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
