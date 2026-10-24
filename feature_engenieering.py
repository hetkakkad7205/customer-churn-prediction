import numpy as np
import pandas as pd

data=pd.read_csv("customer_churn_data.csv")

#Monthly average
data["AverageMonthlySpend"] = np.where(
    df["TenureMonths"] > 0,
    df["TotalCharges"] / df["TenureMonths"],
    df["MonthlyCharges"]
)
print("Average : ",data)

#Contract Grouping
df["ContractGroup"] = df["ContractType"].replace({
    "Month-to-month": "Short-term",
    "Monthly": "Short-term",
    "One year": "Long-term",
    "Two year": "Long-term",
    "1 Year": "Long-term",
    "2 Year": "Long-term"
})

#Remove irrelevant column
data = data.drop(columns=["CustomerID"])
print("Removed",data)

data.to_csv("feature_engineered_churn_data.csv", index=False)

print("\nFeature engineering completed successfully!")
print("Saved as feature_engineered_churn_data.csv")
