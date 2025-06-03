# END-TO-END-DATA-SCIENCE-PROJECT

NAME: RASHMI KUMARI

INTERN ID:CT04DN978

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

TASK DESCRIPTION :
Project Overview:
This project aims to develop a machine learning solution that can predict employee churn using structured HR data. The pipeline includes everything from data preprocessing and feature encoding to model training and deployment using Flask. This solution is based on a real-world dataset and is designed for practical demonstration as part of a Data Science internship.

The dataset used is the IBM HR Analytics Employee Attrition dataset, which contains multiple features such as Age, Department, Monthly Income, Gender, and Years at Company. The target variable is "Attrition," indicating whether an employee has left the company or not. The objective is to build a system that not only performs accurate classification but is also deployable via a lightweight web API.

Step-by-Step Breakdown:

1. Data Loading and Inspection
We start by loading the dataset (data.csv) using the pandas library. This data includes both numerical and categorical variables. Basic inspection is done to understand its structure, including checking for null values, column types, and class distribution of the target variable.

2. Preprocessing
The target column "Attrition" is encoded using LabelEncoder, converting "Yes" to 1 and "No" to 0. Categorical input features such as Department and Gender are converted into numerical format using one-hot encoding (pd.get_dummies). This prepares the data for use in a scikit-learn pipeline.

3. Feature Scaling
Before training the model, features are scaled using StandardScaler. Scaling helps in improving model performance, especially when different features are on different ranges. Scaling is done after splitting the dataset to avoid data leakage.

4. Model Training
A Random Forest Classifier is used as the prediction model. This ensemble method provides good accuracy and handles both categorical and numerical data well. The model is trained on the scaled training set and then evaluated using classification metrics like accuracy, precision, recall, and F1-score.

5. Model Evaluation
The model’s performance is assessed using classification_report from scikit-learn. This provides a detailed breakdown of how well the model predicts employee churn, including metrics for both churned and retained employees.

6. Saving Artifacts
To facilitate deployment, the trained model, the scaler used for transformation, and the list of feature columns are saved using joblib. The files generated are:

model_v2.pkl – Trained RandomForestClassifier

scaler_v2.pkl – Fitted StandardScaler

features_v2.pkl – List of feature column names after encoding

7. Deployment with Flask
A simple Flask application (app_v2.py) is created to serve the model. It provides a POST endpoint (/predict) where a user can send JSON input with employee data. The server will scale the input, align it to the trained feature order, and return the churn prediction in JSON format.

The Flask app uses the saved artifacts and handles unknown inputs gracefully by reindexing and filling missing columns with zero. This ensures robustness of the deployed system.

Conclusion:
This project demonstrates the complete lifecycle of a machine learning pipeline: from raw data to deployment. For internship Task 3, this version not only completes the requirement but also follows good practices such as saving model artifacts, consistent feature handling, and RESTful API deployment. All scripts are modular, and the code is personalized to avoid duplication.

By following this structure, the project is ready for real-world demonstration, academic submission, or GitHub showcase. It highlights a deep understanding of data preprocessing, model training, and deployment — the core skills of a data science intern.
