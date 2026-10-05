"""Détection de fraude – Scoring par Machine Learning.

Usage: python fraud_detection.py [data/creditcard.csv]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
BETA = 2  # recall weighs 2x precision: a missed fraud costs more than a false alert


def add_features(df):
    """Feature matrix from the raw columns (Time, V1..V28, Amount)."""
    X = df.drop(columns=["Class", "Time", "Amount"])
    # Time is seconds since the first transaction, so "hour" is relative to it.
    hour = df["Time"] / 3600 % 24
    X["log_amount"] = np.log1p(df["Amount"])  # Amount is heavily right-skewed
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return X


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=SEED,
        ),
    }


def best_threshold(y, proba, beta=BETA):
    """Decision threshold maximising F-beta along the precision/recall curve."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return thresholds[np.nanargmax(fbeta[:-1])]  # last point has no threshold


def run(df, out_dir="reports"):
    """Train and compare every model with and without SMOTE; write reports to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    X, y = add_features(df), df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    rows, predictions = [], {}
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in make_models().items():
        for smote in (False, True):
            label = f"{name} + SMOTE" if smote else name
            # SMOTE lives inside the pipeline: it only ever sees the training folds.
            steps = [("scale", StandardScaler())]
            if smote:
                steps.append(("smote", SMOTE(random_state=SEED)))
            pipe = Pipeline(steps + [("model", model)])

            # Threshold is tuned on out-of-fold predictions, never on the test set.
            oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            threshold = best_threshold(y_train, oof)

            proba = pipe.fit(X_train, y_train).predict_proba(X_test)[:, 1]
            pred = proba >= threshold
            predictions[label] = pred
            rows.append(
                {
                    "model": label,
                    "pr_auc": average_precision_score(y_test, proba),
                    "roc_auc": roc_auc_score(y_test, proba),
                    "threshold": threshold,
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall": recall_score(y_test, pred),
                    f"f{BETA}": fbeta_score(y_test, pred, beta=BETA),
                }
            )
            PrecisionRecallDisplay.from_predictions(y_test, proba, name=label, ax=ax)
            print(f"done: {label}", file=sys.stderr)

    ax.set_title("Precision-Recall curves (test set)")
    fig.savefig(out_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).round(4)
    results.to_csv(out_dir / "results.csv", index=False)

    best = results.iloc[0]["model"]
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions[best], display_labels=["Legit", "Fraud"], cmap="Blues"
    )
    disp.ax_.set_title(f"{best} (test set)")
    disp.figure_.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(disp.figure_)
    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv"
    print(run(pd.read_csv(path)).to_string(index=False))
