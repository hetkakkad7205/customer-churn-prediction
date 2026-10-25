# 📉 Customer Churn Prediction System

A Data Science and Machine Learning project that predicts whether a customer is likely to leave a service. The project applies data preprocessing, exploratory data analysis, feature engineering, classification algorithms, and model evaluation to support customer retention analysis.

## 📌 Project Overview

Customer churn occurs when customers stop using a company's products or services.

This project analyzes customer characteristics, subscription information, service usage, charges, and support interactions to identify patterns associated with churn.

The system uses classification models to predict whether a customer belongs to the churn or non-churn category.

## ✨ Features

- Customer dataset loading and inspection
- Missing-value handling and data preprocessing
- Exploratory Data Analysis (EDA)
- Customer churn distribution visualizations
- Feature engineering
- Removal of irrelevant identifiers, such as CustomerID
- Classification model training and testing
- Prediction of customer churn
- Model evaluation using classification metrics
- Confusion matrix visualization
- Interactive web interface, if included in the application

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit or Flask, depending on the deployed interface
- HTML, CSS and JavaScript, if using the Flask frontend
- Jupyter Notebook, if used for EDA

## 📂 Project Structure

```text
customer-churn-prediction/
├── app.py
├── model.py
├── data_cleaning.py
├── customer_churn_data.csv
├── cleaned_custumer_churn_data.csv
├── eda_analysis.ipynb
├── requirements.txt
├── README.md
├── templates/
└── static/
    ├── css/
    └── js/
```

*Update filenames and folders to match the actual repository. Check the spelling of the cleaned CSV filename before using it in your code.*

## 🧹 Data Preprocessing and Feature Engineering

The project may include the following steps, depending on the implemented code:

- Handle missing values
- Inspect duplicate records
- Convert categorical variables into model-compatible representations
- Convert the churn target into consistent Yes/No labels or binary values
- Create an average monthly spend feature, where appropriate
- Group contract types into meaningful categories
- Remove irrelevant identifiers such as CustomerID
- Prepare the final dataset for model training

## 🤖 Machine Learning Models

The project explores classification algorithms such as:

- **Random Forest Classifier:** Combines multiple decision trees to improve predictive performance

## 📊 Model Evaluation

The classification models can be evaluated using:

- **Accuracy:** Overall proportion of correct predictions
- **Precision:** Proportion of predicted churn cases that are actually churn cases
- **Recall:** Proportion of actual churn cases correctly identified
- **F1 Score:** Balance between precision and recall
- **Confusion Matrix:** Displays correct and incorrect classifications

Actual metric values should be added after running the evaluation code. Since churn datasets can be imbalanced, accuracy should not be used as the only evaluation metric.

## 🚀 How to Run Locally

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Navigate to the project folder:

   ```bash
   cd customer-churn-prediction
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   For Streamlit:

   ```bash
   streamlit run app.py
   ```

   For Flask, use the appropriate startup command for your application.

## 🔄 Project Workflow

1. Load the customer dataset.
2. Inspect, clean, and preprocess the data.
3. Perform EDA to identify customer patterns.
4. Engineer relevant features.
5. Split the data into training and testing sets.
6. Train the selected classification models.
7. Compare models using evaluation metrics.
8. Predict churn for new customer records.
9. Present results through the web interface, if implemented.

## 🌐 Live Demo

**Live Demo:** YOUR_DEPLOYED_APP_URL

Replace the placeholder with your deployed application URL.

## 👨‍💻 Author

Het Kakkad

## 📄 License

This project is developed for educational and internship purposes.
