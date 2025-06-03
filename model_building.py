import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Data Reading ---
print("Loading dataset...")
df = pd.read_csv('data.csv')

# --- Label encode the target column ---
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes/No to 1/0

# --- Separate features and target variable ---
print("Splitting input and output...")
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# --- Encode categorical values ---
print("Applying one-hot encoding...")
X_encoded = pd.get_dummies(X)

# --- Preserve feature order ---
feature_names = X_encoded.columns.tolist()

# --- Divide into train and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=7)

# --- Scale numeric data ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train classifier ---
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=7)
model.fit(X_train_scaled, y_train)

# --- Evaluate performance ---
y_pred = model.predict(X_test_scaled)
print("\nEvaluation Report:")
print(classification_report(y_test, y_pred))

# --- Store trained components ---
print("Saving model and tools...")
joblib.dump(model, 'model_v2.pkl')
joblib.dump(scaler, 'scaler_v2.pkl')
joblib.dump(feature_names, 'features_v2.pkl')

print("✔️ Training complete. Artifacts saved.")