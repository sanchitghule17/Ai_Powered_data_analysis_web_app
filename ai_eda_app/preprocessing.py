# ai_eda_app/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    

    df = df.copy()

    # 1. Numeric imputation (mean for now – we’ll improve later)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 2. Categorical encoding
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 3. Outlier trimming (z-score < 3)
    z = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)
    df = df[(z < 3).all(axis=1)]

    # 4. Scaling
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    # 5. Dedup
    df.drop_duplicates(inplace=True)

    return df
