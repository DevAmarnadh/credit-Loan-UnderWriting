import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def encode_categorical_variables(data):
    label_encoders = {}
    categorical_columns = ['Gender', 'Married', 'Education', 'Property_Area', 'Self_Employed']
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    # Save the label encoders
    joblib.dump(label_encoders, '../model/label_encoders.pkl')
    
    return data, label_encoders

def add_total_income_feature(data):
    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    return data
