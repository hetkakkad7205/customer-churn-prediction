import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

data=pd.read_csv("cleaned_custumer_churn_data.csv")

# Separate features and target
X = data.drop(columns=["Churn", "CustomerID"])
y = data["Churn"]

# If Churn contains Yes/No, convert it into 1/0
if y.dtype == "object":
    y = y.astype(str).str.strip().str.lower().map({
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0
    })



# Convert categorical input columns to numeric
X = pd.get_dummies(X, drop_first=True)


# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create the model
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=42)

# Train the model FIRST
rf_classifier.fit(X_train, y_train)

print("\nModel training completed!")

#Predict test data AFTER training
y_pred = rf_classifier.predict(X_test)

#Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nMODEL EVALUATION")
print("-----------------------")
print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1 Score:", round(f1, 2))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

#  Visualization: Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

#  Visualization: Model Performance
scores = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

plt.bar(scores.keys(), scores.values())
plt.title("Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

#  Visualization: Feature Importance
importance = pd.Series(
    rf_classifier.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(10)

importance.sort_values().plot(kind="barh")
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

#  Predict 13 test customers
sample = X_test.iloc[:13]
prediction = rf_classifier.predict(sample)

results = pd.DataFrame({
    "Actual Churn": y_test.loc[sample.index],
    "Predicted Churn": prediction
})

results = results.replace({0: "No", 1: "Yes"})

print("\nSample Predictions:")
print(results)






















