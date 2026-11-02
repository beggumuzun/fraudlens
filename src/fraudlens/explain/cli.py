"""Explain the alerts a trained model raises on the test period.

Usage (after ``pip install -e .``)::

    fraudlens-explain --features data/features.csv --model models/fraudlens.joblib

Prints which reason comes first for each attack type (a sanity check: card testing
should be explained by devices and payment bursts, a stolen card by location and
amount) and writes example explanations to reports/explanations.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraudlens.explain.reasons import ReasonExplainer
from fraudlens.features.batch import load_transactions
from fraudlens.model.split import time_split
from fraudlens.model.train import load_bundle, predict_scores


def explain_test_alerts(features: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, list]:
    """Explanations for every test-period alert, plus a top-reason-by-type table."""
    split = time_split(features)
    explainer = ReasonExplainer.from_bundle(bundle, split.train)
    test = split.test
    alerts = test[predict_scores(bundle["model"], test) >= bundle["threshold"]]
    explanations = explainer.explain(alerts)

    top_reason = [e["reasons"][0]["group"] if e["reasons"] else "none" for e in explanations]
    label = alerts["fraud_type"].fillna("false alarm (genuine)")
    table = pd.crosstab(label.to_numpy(), pd.Series(top_reason, name="top reason"))
    table.index.name = "alert type"

    examples = [
        {
            "transaction_id": tid,
            "actual": actual,
            "amount": float(amount),
            **explanation,
        }
        for tid, actual, amount, explanation in zip(
            alerts["transaction_id"], label, alerts["amount"], explanations, strict=True
        )
    ]
    return table, examples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Explain test-period fraud alerts.")
    parser.add_argument("--features", type=Path, default=Path("data/features.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/fraudlens.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/explanations.json"))
    parser.add_argument("--examples", type=int, default=2, help="examples printed per type")
    args = parser.parse_args(argv)

    table, examples = explain_test_alerts(load_transactions(args.features), load_bundle(args.model))
    print("Top reason for each test-period alert:\n")
    print(table.to_string())

    picked, per_type = [], {}
    for example in examples:
        if per_type.get(example["actual"], 0) < args.examples:
            per_type[example["actual"]] = per_type.get(example["actual"], 0) + 1
            picked.append(example)
    for example in picked:
        print(f"\n{example['transaction_id']}  {example['actual']}  score={example['score']:.4f}")
        for reason in example["reasons"]:
            print(f"   {reason['share']:>5.0%}  {reason['text']}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"top_reason_by_type": table.to_dict(orient="index"), "examples": picked}, indent=2
        )
    )


if __name__ == "__main__":
    main()
