import pandas as pd
import numpy as np

data=pd.read_csv("customer_churn_data.csv")

print(data.head())# this will print first top 5 records
print(data.info())# this will print info about datatype
print(data.shape)
