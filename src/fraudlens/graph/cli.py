"""Find fraud rings and their likely point of compromise, and score them against the truth.

Usage (after ``pip install -e .``)::

    fraudlens-rings --transactions data/transactions.csv --features data/features.csv

Uses the saved model's alerts (as a live system would), then writes reports/rings.json.

Two views are reported, because they answer different questions:

* out_of_sample - only alerts on payments the model was NOT trained on (validation and
                  test period). This is the honest measure of what the system would find.
* retrospective - alerts over the whole history. On the training period the model has
                  seen the answers (its recall there is about 100%), so this view is an
                  upper bound, shown for comparison only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraudlens.features.batch import load_transactions
from fraudlens.graph.rings import ring_report, score_against_truth, true_breached_shops
from fraudlens.model.train import load_bundle, predict_scores


def analyse(transactions: pd.DataFrame, alerts: pd.DataFrame) -> dict:
    """Detect rings from ``alerts`` and score them, including the breached shop."""
    groups, compromise = ring_report(transactions, alerts)
    truth = true_breached_shops(transactions)
    ring_of_card = transactions.dropna(subset=["ring_id"]).groupby("card_id")["ring_id"].first()
    shops = []
    for group in (g for g in groups if g.is_ring):
        top = compromise[group.group_id].iloc[0]
        # The real ring behind this group: the one most of its victim cards belong to.
        rings = ring_of_card.reindex(group.cards).dropna()
        real_ring = rings.mode().iat[0] if len(rings) else None
        shops.append(
            {
                "group": group.group_id,
                "cards": group.size,
                "flagged_cards": len(group.flagged_cards),
                "top_shop": top["merchant_id"],
                "victims_who_paid_there": int(top["victims"]),
                "all_customers_share": round(float(top["all_cards_share"]), 3),
                "lift": round(float(top["lift"]), 2),
                "real_ring": real_ring,
                "really_breached": real_ring is not None
                and top["merchant_id"] == truth.get(real_ring),
            }
        )
    return {
        "alerts_used": len(alerts),
        "detection": score_against_truth(groups, transactions, compromise),
        "points_of_compromise": shops,
    }


def _print(name: str, result: dict) -> None:
    score = result["detection"]
    print(
        f"{name}: {score['rings_found']} of {score['true_rings']} rings, "
        f"{score['victim_cards_found']} of {score['victim_cards']} victim cards, "
        f"false ring groups {score['false_ring_groups']}, "
        f"breached shop right for {score['breached_shop_found']} "
        f"({result['alerts_used']} alerts used)"
    )
    for shop in result["points_of_compromise"]:
        check = "correct" if shop["really_breached"] else "WRONG"
        print(
            f"  {shop['group']}: {shop['victims_who_paid_there']} of {shop['flagged_cards']} "
            f"flagged cards paid at {shop['top_shop']} ({shop['lift']}x more than usual), "
            f"{check}"
        )


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
    unseen_from = pd.Timestamp(bundle["split"]["valid"]["from"])

    report = {
        "out_of_sample_from": str(unseen_from.date()),
        "out_of_sample": analyse(transactions, alerts[alerts["timestamp"] >= unseen_from]),
        "retrospective": analyse(transactions, alerts),
    }
    print(f"Linked groups checked: {report['out_of_sample']['detection']['linked_groups_checked']}")
    _print("Out of sample (alerts the model never trained on)", report["out_of_sample"])
    _print("Retrospective (includes training-period alerts)", report["retrospective"])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=bool))


if __name__ == "__main__":
    main()
