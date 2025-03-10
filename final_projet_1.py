"""
Project 1: Controlling the Propagation of a Contagious Disease

This script loads data from 'projet_1_data.csv', trains models (Random Forest and 
Logistic Regression) on both the full feature set and a reduced set (x1, x2), optimizes 
the decision threshold to limit the false negative rate, and plots ROC curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score


def load_data(filepath="projet_1_data.csv"):
    """Load the dataset from CSV."""
    return pd.read_csv(filepath)


def split_data(data, features=None, target='y', test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    if features is None:
        features = ['x1', 'x2', 'x3', 'x4', 'x5']
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_rf(X_train, y_train, random_state=42):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_lr(X_train, y_train, random_state=42, class_weight=None, max_iter=1000, C=1.0):
    """Train a Logistic Regression model."""
    model = LogisticRegression(random_state=random_state, class_weight=class_weight, max_iter=max_iter, C=C)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance and print results."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {model_name} Evaluation ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
    return y_proba


def optimize_threshold(y_true, y_proba, max_fnr=0.1):
    """
    Find the lowest threshold ensuring the false negative rate (FNR) is ≤ max_fnr.
    Returns the optimized threshold.
    """
    best_thresh = 0.5
    best_fpr = 1.0
    for t in np.linspace(0, 1, 101):
        y_pred = (y_proba >= t).astype(int)
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 1.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 1.0
        if fnr <= max_fnr and fpr < best_fpr:
            best_thresh = t
            best_fpr = fpr
    return best_thresh


def plot_roc(y_true, y_proba, model_name="Model"):
    """Plot the ROC curve for a given model."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.2f})")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_models():
    data = load_data()

    # Full Feature Models
    X_train, X_test, y_train, y_test = split_data(data)
    rf_full = train_rf(X_train, y_train)
    rf_proba = evaluate_model(rf_full, X_test, y_test, "Random Forest (Full)")
    rf_thresh = optimize_threshold(y_test, rf_proba)
    print(f"Optimized Threshold for RF (Full): {rf_thresh:.2f}")
    plot_roc(y_test, rf_proba, "Random Forest (Full)")

    lr_full = train_lr(X_train, y_train)
    lr_proba = evaluate_model(lr_full, X_test, y_test, "Logistic Regression (Full)")
    plot_roc(y_test, lr_proba, "Logistic Regression (Full)")

    # Reduced Feature Models (using only x1 and x2)
    features_reduced = ['x1', 'x2']
    X_train_red, X_test_red, y_train, y_test = split_data(data, features=features_reduced)
    rf_red = train_rf(X_train_red, y_train)
    rf_red_proba = evaluate_model(rf_red, X_test_red, y_test, "Random Forest (Reduced)")
    rf_red_thresh = optimize_threshold(y_test, rf_red_proba)
    print(f"Optimized Threshold for RF (Reduced): {rf_red_thresh:.2f}")
    plot_roc(y_test, rf_red_proba, "Random Forest (Reduced)")


def main():
    run_models()


if __name__ == '__main__':
    main()