"""Smoke test on synthetic data shaped like creditcard.csv. Run: python test_fraud_detection.py"""
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from fraud_detection import add_features, best_threshold, run


def synthetic(n=4000):
    X, y = make_classification(
        n_samples=n, n_features=28, n_informative=10, weights=[0.97], random_state=0
    )
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    rng = np.random.default_rng(0)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = rng.exponential(80, n)
    df["Class"] = y
    return df


def test_features():
    X = add_features(synthetic(100))
    assert not {"Class", "Time", "Amount"} & set(X.columns)
    assert X.shape[1] == 31 and X.notna().all().all()
    assert X[["hour_sin", "hour_cos"]].abs().max().max() <= 1


def test_best_threshold():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert best_threshold(y, proba) == 0.8  # perfect separation: lowest score of a positive


def test_pipeline():
    df = synthetic()
    with tempfile.TemporaryDirectory() as tmp:
        results = run(df, tmp)
    assert len(results) == 6
    assert (results["pr_auc"] > df["Class"].mean() * 2).all()  # clearly better than chance
    assert results[["precision", "recall"]].gt(0).all().all()


if __name__ == "__main__":
    test_features()
    test_best_threshold()
    test_pipeline()
    print("ok")
