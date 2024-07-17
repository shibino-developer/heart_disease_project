# src/model_evaluation.py

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data

def evaluate_model(data_path, model_path):
    X, y = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, cm, report

if __name__ == "__main__":
    data_path = '../data/heart.csv'
    model_path = '../model/heart_disease_model.pkl'
    accuracy, cm, report = evaluate_model(data_path, model_path)
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
