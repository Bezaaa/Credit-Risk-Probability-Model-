import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import mlflow
import mlflow.sklearn

def load_data(path="data/processed/high_risk_labels.csv"):
    df = pd.read_csv(path, encoding='latin1')
    return df

def prepare_features(df, target_col="is_high_risk"):
    if target_col in df.columns:
        # Drop target and CustomerId from features
        X = df.drop(columns=[target_col, "CustomerId"])
        y = df[target_col]
        return X, y
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

def main():
    mlflow.set_experiment("credit_risk_model")

    df = load_data()
    X, y = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Feature scaling for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression with Grid Search
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_params = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring="roc_auc", n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)

    # Random Forest with Grid Search
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="roc_auc", n_jobs=-1)
    rf_grid.fit(X_train, y_train)  # Random Forest can work on unscaled data

    # Evaluate models
    models = {
        "LogisticRegression": lr_grid,
        "RandomForest": rf_grid,
    }

    best_model_name = None
    best_auc = 0

    for name, model in models.items():
        X_eval = X_test_scaled if name == "LogisticRegression" else X_test
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\n{name} performance:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Log model and metrics to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_params(model.best_params_)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.sklearn.log_model(model.best_estimator_, f"{name}_model")

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_name = name

    print(f"\nBest model: {best_model_name} with ROC AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
