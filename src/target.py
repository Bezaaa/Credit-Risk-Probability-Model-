import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max()

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "TransactionId": "count",
        "Value": "sum"
    }).reset_index()

    rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]
    return rfm

def cluster_rfm(rfm_df: pd.DataFrame, n_clusters=3, random_state=42) -> pd.DataFrame:
    rfm_scaled = StandardScaler().fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df["Cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm_df

def assign_high_risk_label(rfm_df: pd.DataFrame) -> pd.DataFrame:
    
    cluster_summary = rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = cluster_summary.sort_values(by=["Frequency", "Monetary", "Recency"], ascending=[True, True, False]).index[0]

    rfm_df["is_high_risk"] = (rfm_df["Cluster"] == high_risk_cluster).astype(int)
    return rfm_df[["CustomerId", "is_high_risk"]]

def generate_proxy_labels(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    rfm = calculate_rfm(df)
    clustered_rfm = cluster_rfm(rfm)
    rfm_with_label = assign_high_risk_label(clustered_rfm)

    if output_csv:
        rfm_with_label.to_csv(output_csv, index=False)

    return rfm_with_label
if __name__ == "__main__":
    labels = generate_proxy_labels("data/raw/data.csv", "data/processed/high_risk_labels.csv")
    print(labels.head())
