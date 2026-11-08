import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer


def prepare_data(df):
    df = df.copy()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    return X, y

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['str']).columns.tolist()  

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(transformers=[
        ('nums', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])