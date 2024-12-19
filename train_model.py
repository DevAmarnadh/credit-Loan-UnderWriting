import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Verify and update file path
file_path = './data/train_cleaned.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at {file_path}")
else:
    print(f"File found at {file_path}")

# Load training data
train_data = pd.read_csv(file_path)

# Identify categorical columns and numeric columns
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Encode categorical variables
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])
    label_encoders[column] = le

# Encode target variable
le_status = LabelEncoder()
train_data['Loan_Status'] = le_status.fit_transform(train_data['Loan_Status'])

# Separate features and target
X = train_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = train_data['Loan_Status']

# Standardize numeric variables
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Output class distribution
print(f"Class distribution in training set: {np.bincount(y)}")

# Save the model
model_save_path = './model/random_forest_model.pkl'
joblib.dump(model, model_save_path)
print(f"Model saved at {model_save_path}")

# Save the label encoders and scaler
encoders_save_path = './model/label_encoders.pkl'
joblib.dump(label_encoders, encoders_save_path)
print(f"Label encoders saved at {encoders_save_path}")

scaler_save_path = './model/scaler.pkl'
joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved at {scaler_save_path}")

# Save the label encoder for the target variable
status_encoder_save_path = './model/status_encoder.pkl'
joblib.dump(le_status, status_encoder_save_path)
print(f"Status encoder saved at {status_encoder_save_path}")
