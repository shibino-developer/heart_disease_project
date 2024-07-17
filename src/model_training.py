# src/model_training.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from data_preprocessing import load_and_preprocess_data


def train_model(data_path, model_path):
    X, y = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model

if __name__ == "__main__":
    data_path = 'data/heart.csv'
    print(f"Current working directory: {os.getcwd()}")
    print(f"Absolute data path: {os.path.abspath(data_path)}")
    print(f"Exists: {os.path.exists(data_path)}")  # Check if file exists
    model_path = 'model/heart_disease_model.pkl'
    train_model(data_path, model_path)
