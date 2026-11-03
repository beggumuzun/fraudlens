"""Find fraud rings and their likely point of compromise, and score them against the truth.

Usage (after ``pip install -e .``)::

    fraudlens-rings --transactions data/transactions.csv --features data/features.csv

Uses the saved model's alerts (as a live system would), then writes reports/rings.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraudlens.features.batch import load_transactions
from fraudlens.graph.rings import ring_report, score_against_truth
from fraudlens.model.train import load_bundle, predict_scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect fraud rings from model alerts.")
    parser.add_argument("--transactions", type=Path, default=Path("data/transactions.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/fraudlens.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/rings.json"))
    args = parser.parse_args(argv)

    transactions = load_transactions(args.transactions)
    features = load_transactions(args.features)
    bundle = load_bundle(args.model)
    flagged = predict_scores(bundle["model"], features) >= bundle["threshold"]
    alerts = transactions.loc[flagged, ["card_id", "timestamp"]]

    groups, compromise = ring_report(transactions, alerts)
    score = score_against_truth(groups, transactions)
    rings = [g for g in groups if g.is_ring]
    shops = []
    for group in rings:
        top = compromise[group.group_id].iloc[0]
        shops.append(
            {
                "group": group.group_id,
                "cards": group.size,
                "flagged_cards": len(group.flagged_cards),
                "top_shop": top["merchant_id"],
                "victims_who_paid_there": int(top["victims"]),
                "all_customers_share": round(float(top["all_cards_share"]), 3),
                "lift": round(float(top["lift"]), 2),
            }
        )

    print(f"Linked groups checked: {score['linked_groups_checked']}")
    print(
        f"Rings found: {score['rings_found']} of {score['true_rings']} "
        f"({score['victim_cards_found']} of {score['victim_cards']} victim cards), "
        f"false ring groups: {score['false_ring_groups']}"
    )
    for shop in shops:
        print(
            f"  {shop['group']}: {shop['victims_who_paid_there']} of {shop['flagged_cards']} "
            f"flagged cards paid at {shop['top_shop']} ({shop['lift']}x more than usual)"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"detection": score, "points_of_compromise": shops}, indent=2)
    )


if __name__ == "__main__":
    main()
