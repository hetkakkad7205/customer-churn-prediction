import pandas as pd
import numpy as np

data=pd.read_csv("customer_churn_data.csv")

print(data.head())

#removes extra spaces
data.columns=data.columns.str.strip()

#removes duplicates
data=data.drop_duplicates()

#check missing values
print("Missing Values: ")
print(data.isnull().sum())

#convert numeric columns to numeric dt :- if "20" then 20
nums_cols=["Age","TenureMonths","MonthlyCharges","TotalCharges","SupportCalls"]
for col in nums_cols:
    data[col]=pd.to_numeric(data[col])

# clean the Churn target column
data["Churn"] = (data["Churn"].astype("string").str.strip().str.lower())

data["Churn"] = data["Churn"].map({
    "yes": "Yes",
    "no": "No",
    "1": "Yes",
    "0": "No",
    "true": "Yes",
    "false": "No"
})
data=data.dropna(subset=["Churn"])

#handle missing numircal columns
for col in nums_cols:
    data[col] = data[col].fillna(0)

# Handle missing categorical values using mode
categorical_columns = [
    "Gender",
    "Subscription",
    "ContractType",
    "InternetService",
    "PhoneService",
    "StreamingService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod"
]

print("Cleaned Dataset: ")
print(data.head())
print("Cleaned Dataset Shape: ",data.shape)

print("Churn Distribution: ",data["Churn"].value_counts())
print("Saved Cleaned csv file")

#saves cleaned csv file
data.to_csv("cleaned_custumer_churn_data.csv", index=False)
