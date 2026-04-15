"""
Model training script for MLOps project
Trains a classifier on the Iris dataset and saves the model
"""

import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import os

# Configuration
MODEL_PATH = "model.pkl"
DATA_DIR = "data"

def load_data():
    """Load Iris dataset"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y

def preprocess_data(X, y):
    """Simple preprocessing - normalize features"""
    X_processed = (X - X.mean()) / X.std()
    return X_processed, y

def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50 + "\n")

    return {"accuracy": accuracy, "precision": precision, "f1": f1}

def save_model(model, path=MODEL_PATH):
    """Save model to disk"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {path}")

def main():
    print("Starting model training pipeline...")

    # Load data
    print("Loading Iris dataset...")
    X, y = load_data()
    print(f"✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")

    # Preprocess
    print("Preprocessing data...")
    X_processed, y = preprocess_data(X, y)
    print("✓ Data preprocessed (normalized)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    print(f"✓ Split data: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # Train model
    print("Training Random Forest classifier...")
    model = train_model(X_train, y_train)
    print("✓ Model trained")

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)

    print("✓ Training pipeline completed successfully!")
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
