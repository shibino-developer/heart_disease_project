# src/data_preprocessing.py

import pandas as pd

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y
