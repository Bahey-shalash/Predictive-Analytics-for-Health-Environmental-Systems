"""
Project 2: Analyse d’un phénomène environnemental inconnu

This script loads environmental sensor data from 'projet_2_data.csv', applies feature
transformations (temporal features, cyclic transformations, and lag features), splits
the data into training and test sets based on a cutoff date, trains a RandomForestRegressor
model on the training set, evaluates its performance on both sets, and plots the results.
Finally, it runs a verification (if available) using eng209.verify.verify_q2.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit  # not used explicitly, but useful for time series

# ---------------------
# Parameters and Paths
# ---------------------
INPUT_FILENAME = './projet_2_data.csv'
HOLD_OUT_FILENAME = './projet_2_hold_out.csv'
SPLIT_DATE = pd.Timestamp("2024-02-19 12:00:00")  # Cutoff for training/test split

# ----------------------------
# Data Transformation Function
# ----------------------------
def inputTransform(file_path: str = HOLD_OUT_FILENAME):
    """
    Transforms raw CSV data into features, target (if available), and timestamps.
    
    The function:
      - Parses the timestamp column and sets it as the DataFrame index.
      - Extracts temporal features: hour, weekday, sin_hour, cos_hour.
      - Creates lag features on x1 and x2 (15 and 30 minutes, given a 15-min interval).
      - Drops rows with missing values.
    
    Returns:
      X: DataFrame of features.
      y: Series of target values if present, else None.
      timestamps: Index (timestamps) of the data.
    """
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    
    # Temporal features
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Lag features: one lag = 15 minutes; two lags = 30 minutes
    df['x1_lag15'] = df['x1'].shift(1)
    df['x1_lag30'] = df['x1'].shift(2)
    df['x2_lag15'] = df['x2'].shift(1)
    df['x2_lag30'] = df['x2'].shift(2)
    
    # Drop rows with NaN (introduced by lag features)
    df = df.dropna()
    
    features = ['x1', 'x2', 'x1_lag15', 'x1_lag30', 'x2_lag15', 'x2_lag30', 'sin_hour', 'cos_hour', 'weekday']
    X = df[features]
    y = df['y'] if 'y' in df.columns else None
    timestamps = df.index
    
    return X, y, timestamps

# --------------------------
# Data Splitting Function
# --------------------------
def split_train_test(X, y, timestamps, split_date=SPLIT_DATE):
    """
    Splits the dataset into training (before split_date) and test (on/after split_date) sets.
    
    Returns:
      X_train, y_train: Data before split_date.
      X_test, y_test: Data on or after split_date.
    """
    train_mask = timestamps < split_date
    test_mask = timestamps >= split_date
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    return X_train, y_train, X_test, y_test

# --------------------------
# Model Evaluation Functions
# --------------------------
def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluates a regression model by computing R², MSE, RMSE, MAE, and MAPE.
    Prints the metrics and returns the model predictions.
    """
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    print(f"{dataset_name} Evaluation:")
    print("R²      :", r2)
    print("MSE     :", mse)
    print("RMSE    :", rmse)
    print("MAE     :", mae)
    print("MAPE(%) :", mape)
    return y_pred

def plot_predictions(timestamps, y_true, y_pred, title="Predictions vs True Values"):
    """
    Plots true vs. predicted values over time.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, y_true, label="True Values", alpha=0.7)
    plt.plot(timestamps, y_pred, label="Predicted Values", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_scatter(y_true, y_pred, title="Scatter: True vs Predicted"):
    """
    Plots a scatter plot of true vs. predicted values.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals(timestamps, residuals, title="Residuals over Time"):
    """
    Plots the distribution of residuals and their trend over time.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    plt.title("Residuals Distribution")
    plt.xlabel("Residual (y - predicted)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------
# Main Execution Function
# --------------------------
def main():
    # Transform the full dataset (with target y)
    X_full, y_full, ts_full = inputTransform(INPUT_FILENAME)
    print("Full data dimensions:", X_full.shape, y_full.shape)
    
    # Split the data based on the split date
    X_train, y_train, X_test, y_test = split_train_test(X_full, y_full, ts_full, SPLIT_DATE)
    print("Training set:", X_train.shape, y_train.shape)
    print("Test set:", X_test.shape, y_test.shape)
    
    # Train the RandomForestRegressor on the training set
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model performance on the training set
    print("\n--- Training Set Evaluation ---")
    y_train_pred = evaluate_model(model, X_train, y_train, "Training")
    
    # Evaluate model performance on the test set
    print("\n--- Test Set Evaluation ---")
    y_test_pred = evaluate_model(model, X_test, y_test, "Test")
    
    # Plot predictions vs. true values on the test set
    plot_predictions(X_test.index, y_test, y_test_pred, "Test Set: True vs Predicted")
    plot_scatter(y_test, y_test_pred, "Test Set: True vs Predicted Scatter")
    
    # Plot residual analysis on the test set
    residuals_test = y_test - y_test_pred
    plot_residuals(X_test.index, residuals_test, "Test Set Residuals over Time")
    
    # Verification: Check model submission if the module is available
    try:
        from eng209.verify import verify_q2
        verify_q2(model, inputGenerator=inputTransform)
    except ImportError:
        print("Verification module not found. Skipping verification.")

if __name__ == '__main__':
    main()