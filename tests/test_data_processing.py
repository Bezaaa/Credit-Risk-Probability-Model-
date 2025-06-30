import pandas as pd
import pytest
from src.train import load_data, prepare_features

def test_load_data():
    df = load_data("../data/processed/high_risk_labels.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_prepare_features_success():
    df = pd.DataFrame({
        "feature1": [0.1, 0.2, 0.3],
        "is_high_risk": [0, 1, 0]
    })
    X, y = prepare_features(df, target_col="is_high_risk")
    assert len(X) == len(y)
    assert "is_high_risk" not in X.columns

def test_prepare_features_missing_target():
    df = pd.DataFrame({"feature1": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError):
        prepare_features(df, target_col="is_high_risk")
