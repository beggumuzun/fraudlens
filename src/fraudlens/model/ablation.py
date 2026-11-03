"""Ablation: does a feature really help? Retrain without it and compare on the test set.

The question here is the network feature ``device_card_count`` (how many other cards
used this device before). It looks like the perfect fraud-ring signal, but every ring
payment is ALSO from a brand-new device, which the model already knows about. So:
retrain the same gradient-boosting model with the feature switched off (set to 0
everywhere) and compare, above all on ring attacks.

Usage (after ``pip install -e .``)::

    fraudlens-ablation --features data/features.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraudlens.features.batch import load_transactions
from fraudlens.features.definitions import FEATURE_COLUMNS
from fraudlens.model.metrics import evaluate, threshold_at_fpr
from fraudlens.model.split import time_split
from fraudlens.model.train import DEFAULT_MAX_FPR, make_gradient_boosting, predict_scores

ABLATED_FEATURE = "device_card_count"


def ring_attack_recall(scored: pd.DataFrame, threshold: float) -> dict:
    """For ring attacks: share caught at all, and share caught on their FIRST payment."""
    rings = scored[scored["ring_id"].notna()].sort_values("timestamp")
    if rings.empty:
        return {"ring_attacks": 0, "caught": None, "caught_on_first_payment": None}
    per_card = rings.groupby("card_id")
    caught = per_card["score"].max() >= threshold
    first = per_card["score"].first() >= threshold
    return {
        "ring_attacks": int(len(caught)),
        "caught": round(float(caught.mean()), 4),
        "caught_on_first_payment": round(float(first.mean()), 4),
    }


def run(
    features: pd.DataFrame,
    feature: str = ABLATED_FEATURE,
    max_fpr: float = DEFAULT_MAX_FPR,
    seed: int = 42,
) -> dict:
    """Train with and without ``feature``; return test metrics for both."""
    if feature not in FEATURE_COLUMNS:
        raise ValueError(f"Unknown feature: {feature}")
    split = time_split(features)
    results = {}
    for name, switch_off in (("with_feature", False), ("without_feature", True)):

        def prepare(part: pd.DataFrame, switch_off: bool = switch_off) -> pd.DataFrame:
            return part.assign(**{feature: 0.0}) if switch_off else part

        model = make_gradient_boosting(seed)
        model.fit(prepare(split.train)[FEATURE_COLUMNS], split.train["is_fraud"])
        threshold = threshold_at_fpr(
            split.valid["is_fraud"], predict_scores(model, prepare(split.valid)), max_fpr
        )
        test = split.test
        scored = test[["card_id", "timestamp", "amount", "is_fraud", "fraud_type", "ring_id"]]
        scored = scored.assign(score=predict_scores(model, prepare(test)))
        metrics = evaluate(scored, threshold)
        results[name] = {
            "average_precision": round(metrics["average_precision"], 4),
            "recall": round(metrics["recall"], 4),
            "attack_recall": round(metrics["attack_recall"], 4),
            "false_positive_rate": round(metrics["false_positive_rate"], 5),
            **ring_attack_recall(scored, threshold),
        }
    return {"feature": feature, "max_fpr": max_fpr, **results}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retrain without one feature and compare.")
    parser.add_argument("--features", type=Path, default=Path("data/features.csv"))
    parser.add_argument("--feature", default=ABLATED_FEATURE)
    parser.add_argument("--report", type=Path, default=Path("reports/ablation.json"))
    args = parser.parse_args(argv)

    result = run(load_transactions(args.features), feature=args.feature)
    table = pd.DataFrame({k: v for k, v in result.items() if isinstance(v, dict)})
    print(f"Test set, gradient boosting, with and without {args.feature}\n")
    print(table.to_string())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
