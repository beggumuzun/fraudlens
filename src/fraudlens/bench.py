"""Latency benchmark: how long each live step takes for ONE payment.

Measures, on the saved model and data:

* features - the online feature engine (``OnlineFeatureStore.process``)
* score    - the model scoring one payment
* explain  - reason codes for one flagged payment (256 coalitions, one model call)

Usage (after ``pip install -e .``)::

    fraudlens-bench --transactions data/transactions.csv --features data/features.csv

Numbers depend on the machine; reports/benchmark.json records where they came from.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from fraudlens.data.generator import iter_transactions
from fraudlens.explain.reasons import ReasonExplainer
from fraudlens.features.batch import load_transactions
from fraudlens.features.definitions import FEATURE_COLUMNS
from fraudlens.features.online import OnlineFeatureStore
from fraudlens.model.train import load_bundle, predict_scores


def _summary(timings_ms: list[float]) -> dict:
    return {
        "runs": len(timings_ms),
        "p50_ms": round(float(np.percentile(timings_ms, 50)), 3),
        "p99_ms": round(float(np.percentile(timings_ms, 99)), 3),
    }


def _timed(fn, items) -> list[float]:
    timings = []
    for item in items:
        start = time.perf_counter()
        fn(item)
        timings.append((time.perf_counter() - start) * 1_000)
    return timings


def run(
    transactions: pd.DataFrame,
    features: pd.DataFrame,
    bundle: dict,
    n_payments: int = 5_000,
    n_explain: int = 200,
) -> dict:
    """Time the three live steps; the feature engine is warmed on all earlier history."""
    store = OnlineFeatureStore()
    warm = transactions.iloc[:-n_payments]
    timed = transactions.iloc[-n_payments:]
    for txn in iter_transactions(warm):
        store.update(txn)
    feature_times = _timed(store.process, list(iter_transactions(timed)))

    model = bundle["model"]
    last = features.tail(1_000)[FEATURE_COLUMNS]
    rows = [last.iloc[[i]] for i in range(len(last))]
    score_times = _timed(model.predict_proba, rows)

    explainer = ReasonExplainer.from_bundle(bundle)
    flagged = features[predict_scores(model, features) >= bundle["threshold"]].tail(n_explain)
    explain_times = _timed(explainer.explain, [flagged.iloc[[i]] for i in range(len(flagged))])

    return {
        "features": _summary(feature_times),
        "score": _summary(score_times),
        "explain": _summary(explain_times),
        "model": bundle["model_name"],
        "machine": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "processor": platform.processor() or platform.machine(),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Time the live scoring steps.")
    parser.add_argument("--transactions", type=Path, default=Path("data/transactions.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/fraudlens.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args(argv)

    transactions = load_transactions(args.transactions)
    result = run(transactions, load_transactions(args.features), load_bundle(args.model))
    for step in ("features", "score", "explain"):
        r = result[step]
        print(f"{step:>8}: p50 {r['p50_ms']} ms, p99 {r['p99_ms']} ms ({r['runs']} runs)")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
