"""Training script for Linear Regression models"""

from main import LinearRegressionModel
import pandas as pd
import numpy as np

def train_linear_regression(data_path, target_column):
    """
    Complete training pipeline for linear regression
    
    Args:
        data_path: Path to training data CSV
        target_column: Name of target column
    
    Returns:
        Trained model and evaluation metrics
    """
    
    # Initialize model
    model = LinearRegressionModel()
    
    # Load data
    df = model.load_data(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Prepare data (split + scale)
    X_train, X_test, y_train, y_test = model.prepare_data(X, y)
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate
    metrics, predictions = model.evaluate(X_test, y_test)
    
    print("\n=== Model Evaluation ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return model, metrics, predictions


if __name__ == "__main__":
    print("Linear Regression Training Script")
    print("Modify data_path and target_column to use your dataset")
