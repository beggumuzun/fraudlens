"""How we judge a fraud model.

Accuracy is useless here: a model that never flags anything is 99.7% accurate.
Banks think in terms of an *alert budget*: analysts can only review so many alerts,
and every false alarm annoys a genuine customer whose card gets blocked. So we:

1. fix the false-positive rate (e.g. at most 1 false alarm per 1,000 genuine payments),
2. pick the score threshold that respects that budget on the validation set,
3. report how much fraud is caught at that threshold, by count, by money and by attack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y_true, scores, max_fpr: float) -> float:
    """Lowest threshold (flag when score >= threshold) whose false-positive rate <= max_fpr."""
    y_true = np.asarray(y_true)
    genuine = np.sort(np.asarray(scores, dtype=float)[y_true == 0])
    if len(genuine) == 0:
        raise ValueError("Need at least one genuine payment to set a threshold")
    allowed_false_alarms = int(np.floor(max_fpr * len(genuine)))
    if allowed_false_alarms >= len(genuine):
        return float("-inf")
    # Everything strictly above this genuine score gets flagged.
    cutoff = genuine[len(genuine) - allowed_false_alarms - 1]
    return float(np.nextafter(cutoff, np.inf))


def evaluate(scored: pd.DataFrame, threshold: float) -> dict:
    """Metrics for a frame with columns: score, is_fraud, fraud_type, amount, card_id."""
    y = scored["is_fraud"].to_numpy()
    scores = scored["score"].to_numpy()
    flagged = scores >= threshold
    fraud = y == 1
    genuine = ~fraud

    true_alerts = int((flagged & fraud).sum())
    false_alerts = int((flagged & genuine).sum())

    # Money: what share of stolen pounds would have been stopped?
    fraud_amount = scored.loc[fraud, "amount"].sum()
    caught_amount = scored.loc[flagged & fraud, "amount"].sum()

    # Attacks: one attack = one card's fraud episode. Caught if any of its payments is flagged.
    attacks = scored[fraud].assign(flagged=flagged[fraud]).groupby(["card_id", "fraud_type"])
    attack_caught = attacks["flagged"].any()

    recall_by_type = (
        scored[fraud].assign(flagged=flagged[fraud]).groupby("fraud_type")["flagged"].mean()
    )

    return {
        "n_payments": int(len(y)),
        "n_frauds": int(fraud.sum()),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold": float(threshold),
        "alerts": int(flagged.sum()),
        "false_positive_rate": false_alerts / max(int(genuine.sum()), 1),
        "precision": true_alerts / max(int(flagged.sum()), 1),
        "recall": true_alerts / max(int(fraud.sum()), 1),
        "value_recall": float(caught_amount / fraud_amount) if fraud_amount else 0.0,
        "attacks": int(len(attack_caught)),
        "attack_recall": float(attack_caught.mean()) if len(attack_caught) else 0.0,
        "recall_by_type": {k: float(v) for k, v in recall_by_type.items()},
    }
