"""Train, compare and save fraud models.

Two candidates are trained on the same data:

* logistic_regression - simple, linear, easy to explain: the baseline to beat
* gradient_boosting   - many small decision trees that can learn combinations such as
                        "far from home AND at night AND on a new device"

The one with the best validation score is saved together with its alert threshold.

Usage (after ``pip install -e .``)::

    fraudlens-train --features data/features.csv
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from fraudlens import __version__
from fraudlens.features.batch import load_transactions
from fraudlens.features.definitions import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from fraudlens.model.metrics import evaluate, threshold_at_fpr
from fraudlens.model.split import time_split

DEFAULT_MAX_FPR = 0.001  # at most 1 false alarm per 1,000 genuine payments


def make_logistic_regression(seed: int) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    # class_weight="balanced": fraud is rare, so each fraud counts ~300x more in the loss.
    clf = LogisticRegression(class_weight="balanced", max_iter=5_000, random_state=seed)
    return Pipeline([("prep", prep), ("clf", clf)])


def make_gradient_boosting(seed: int) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=[len(NUMERIC_FEATURES)],  # merchant_category comes last
        early_stopping=False,
        random_state=seed,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


MODELS = {
    "logistic_regression": make_logistic_regression,
    "gradient_boosting": make_gradient_boosting,
}


def predict_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Fraud score between 0 and 1 for each row."""
    return model.predict_proba(features[FEATURE_COLUMNS])[:, 1]


def _scored(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    return df[["card_id", "amount", "is_fraud", "fraud_type"]].assign(score=scores)


def _single_payment_latency_ms(model: Pipeline, rows: pd.DataFrame, n: int = 200) -> dict:
    """How long scoring ONE payment takes, as the live API will do it."""
    timings = []
    for i in range(min(n, len(rows))):
        row = rows.iloc[[i]][FEATURE_COLUMNS]
        start = time.perf_counter()
        model.predict_proba(row)
        timings.append((time.perf_counter() - start) * 1_000)
    return {"p50": float(np.percentile(timings, 50)), "p99": float(np.percentile(timings, 99))}


def train(
    features: pd.DataFrame,
    max_fpr: float = DEFAULT_MAX_FPR,
    seed: int = 42,
    valid_frac: float = 0.15,
    test_frac: float = 0.2,
) -> dict:
    """Train every candidate, pick the best on validation, return a saveable bundle."""
    split = time_split(features, valid_frac=valid_frac, test_frac=test_frac)
    if split.train["is_fraud"].sum() == 0 or split.valid["is_fraud"].sum() == 0:
        raise ValueError("Train and validation sets both need fraud examples")

    candidates = {}
    for name, factory in MODELS.items():
        model = factory(seed)
        start = time.perf_counter()
        model.fit(split.train[FEATURE_COLUMNS], split.train["is_fraud"])
        fit_seconds = time.perf_counter() - start

        valid_scores = predict_scores(model, split.valid)
        threshold = threshold_at_fpr(split.valid["is_fraud"], valid_scores, max_fpr)
        candidates[name] = {
            "model": model,
            "threshold": threshold,
            "fit_seconds": round(fit_seconds, 2),
            "valid": evaluate(_scored(split.valid, valid_scores), threshold),
            "test": evaluate(_scored(split.test, predict_scores(model, split.test)), threshold),
        }

    best_name = max(candidates, key=lambda n: candidates[n]["valid"]["average_precision"])
    best = candidates[best_name]

    importance = permutation_importance(
        best["model"],
        split.valid[FEATURE_COLUMNS],
        split.valid["is_fraud"],
        scoring="average_precision",
        n_repeats=5,
        random_state=seed,
    )
    feature_importance = dict(
        sorted(
            zip(FEATURE_COLUMNS, importance.importances_mean.round(4).tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    return {
        "model": best["model"],
        "model_name": best_name,
        "threshold": best["threshold"],
        "max_fpr": max_fpr,
        "feature_columns": FEATURE_COLUMNS,
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": split.describe(),
        "candidates": {
            name: {k: v for k, v in c.items() if k != "model"} for name, c in candidates.items()
        },
        "feature_importance": feature_importance,
        "latency_ms": _single_payment_latency_ms(best["model"], split.test),
    }


def report(bundle: dict) -> dict:
    """Everything in the bundle except the model object itself (JSON friendly)."""
    return {k: v for k, v in bundle.items() if k != "model"}


def save_bundle(bundle: dict, model_path: Path, report_path: Path | None = None) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report(bundle), indent=2))


def load_bundle(model_path: Path) -> dict:
    return joblib.load(model_path)


def _print_summary(bundle: dict) -> None:
    rows = {
        name: {
            "PR-AUC": c["test"]["average_precision"],
            "recall": c["test"]["recall"],
            "precision": c["test"]["precision"],
            "£ recall": c["test"]["value_recall"],
            "attacks caught": c["test"]["attack_recall"],
            "FPR": c["test"]["false_positive_rate"],
        }
        for name, c in bundle["candidates"].items()
    }
    print(f"Test set, alert threshold set for FPR <= {bundle['max_fpr']:.2%} on validation\n")
    print(pd.DataFrame(rows).T.round(4).to_string())
    best = bundle["candidates"][bundle["model_name"]]["test"]
    print(f"\nSaved model: {bundle['model_name']}")
    print("Recall by attack type:", {k: round(v, 3) for k, v in best["recall_by_type"].items()})
    print("Top features:", list(bundle["feature_importance"])[:5])
    print(f"Single-payment latency: {bundle['latency_ms']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare fraud models.")
    parser.add_argument("--features", type=Path, default=Path("data/features.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/fraudlens.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--max-fpr", type=float, default=DEFAULT_MAX_FPR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    features = load_transactions(args.features)
    bundle = train(features, max_fpr=args.max_fpr, seed=args.seed)
    save_bundle(bundle, args.out, args.report)
    _print_summary(bundle)


if __name__ == "__main__":
    main()
