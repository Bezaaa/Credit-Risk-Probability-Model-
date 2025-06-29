import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class RFMFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")

        if self.snapshot_date is None:
            self.snapshot_date = df["TransactionStartTime"].max()

        rfm = df.groupby("CustomerId").agg({
            "TransactionStartTime": lambda x: (self.snapshot_date - x.max()).days,
            "TransactionId": "count",
            "Value": "sum"
        }).reset_index()

        rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]
        return rfm


def build_pipeline():
    numeric_features = ["Recency", "Frequency", "Monetary"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_features = []
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[
        ("rfm", RFMFeatureGenerator()),
        ("preprocessor", preprocessor)
    ])

    return pipeline


def process_and_save_features(input_path: str, output_path: str = None):
    df = pd.read_csv(input_path)
    pipeline = build_pipeline()
    features = pipeline.fit_transform(df)

    if output_path:
        np.save(output_path, features)

    return features


def main():
    input_path = "data/raw/data.csv"  
    output_path = "data/processed/processed_data.csv"
 

    print("Starting data processing...")
    processed_features = process_and_save_features(input_path, output_path)
    print(f"Processed features shape: {processed_features.shape}")

  
    feature_names = ["Recency", "Frequency", "Monetary"]  
    df_features = pd.DataFrame(processed_features, columns=feature_names)
    print(df_features.head())


if __name__ == "__main__":
    main()
