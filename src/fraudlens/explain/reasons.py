"""Reason codes: a plain-English "why" for every fraud score.

Method: exact Shapley values over groups of features.

1. Features are grouped by meaning (location, velocity, amount, ...): 7 groups.
2. A *typical genuine payment* (medians from the training data) is the reference.
3. The payment is re-scored for all 2^7 = 128 mixes of "this group has its real value"
   and "this group looks typical".
4. Each group gets its Shapley value: its average extra contribution over all those
   mixes. The values add up exactly to (score of this payment - score of a typical one).
5. The groups with the largest share are turned into sentences using the real values.

Why not simply switch off one group at a time? Fraud is an AND of signals ("abroad AND
a new device"). Making either one look normal drops the score to almost zero, so every
group would look 99% responsible. Shapley values share the credit fairly instead.

Why groups instead of single features? Related features carry the same signal ("outside
the UK" and "4,800 km from home"); split apart, each would get only part of the credit.

Scores are compared in log-odds, the model's own internal scale, where contributions
add up more naturally than in probabilities. All 128 variants of all payments are
scored in one model call: explaining one payment takes a few milliseconds.
"""

from __future__ import annotations

from itertools import product
from math import factorial

import numpy as np
import pandas as pd

from fraudlens.features.definitions import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

FEATURE_GROUPS: dict[str, list[str]] = {
    "location": ["dist_from_usual_km", "dist_from_last_km", "speed_from_last_kmh", "is_outside_uk"],
    "velocity": [
        "seconds_since_last",
        "txn_count_10m",
        "txn_count_1h",
        "txn_count_24h",
        "amount_sum_1h",
        "amount_sum_24h",
        "small_txn_count_10m",
    ],
    "amount": ["amount", "log_amount", "amount_to_avg_ratio", "amount_zscore"],
    "device": ["is_new_device", "new_device_24h", "is_online", "device_card_count"],
    "timing": ["hour", "is_night", "hour_familiarity"],
    "merchant": ["merchant_category", "is_new_category"],
    "history": ["n_prev_txns"],
}
MIN_SHARE = 0.10  # only report groups behind at least 10% of the extra suspicion
IMPOSSIBLE_SPEED_KMH = 900  # faster than a passenger jet

# Row b of MASKS says which groups keep their real value in coalition b (bit g = group g).
_N_GROUPS = len(FEATURE_GROUPS)
MASKS = np.array(list(product([0, 1], repeat=_N_GROUPS)))[:, ::-1].astype(bool)


def _shapley_weights(n: int) -> np.ndarray:
    """Weight of a coalition of size s: s! (n - s - 1)! / n!"""
    return np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])


_WEIGHTS = _shapley_weights(_N_GROUPS)


def reference_profile(features: pd.DataFrame) -> dict:
    """A 'typical genuine payment': median of each numeric feature, most common category."""
    genuine = features[features["is_fraud"] == 0] if "is_fraud" in features else features
    reference = {col: float(genuine[col].median()) for col in NUMERIC_FEATURES}
    for col in CATEGORICAL_FEATURES:
        reference[col] = str(genuine[col].mode().iat[0])
    return reference


def _payments(count: float) -> str:
    return f"{count:.0f} payment" + ("" if count == 1 else "s")


def describe(group: str, f: pd.Series) -> str:
    """Turn one feature group of one payment into a sentence an analyst can act on."""
    if group == "location":
        if f["speed_from_last_kmh"] > IMPOSSIBLE_SPEED_KMH:
            return (
                f"Impossible travel: {f['dist_from_last_km']:,.0f} km from the previous "
                f"payment, a speed of {f['speed_from_last_kmh']:,.0f} km/h"
            )
        where = "outside the UK, " if f["is_outside_uk"] else ""
        return f"Paid {where}{f['dist_from_usual_km']:,.0f} km from the card's usual area"
    if group == "velocity":
        if f["small_txn_count_10m"] >= 2:
            return (
                f"{f['small_txn_count_10m']:.0f} tiny payments (under £5) in the previous "
                "10 minutes, a typical card-testing pattern"
            )
        if f["txn_count_10m"] >= 1:
            return (
                f"{_payments(f['txn_count_10m'])} in the previous 10 minutes, "
                f"£{f['amount_sum_1h']:,.0f} spent in the last hour"
            )
        return (
            f"{_payments(f['txn_count_1h'])} and £{f['amount_sum_1h']:,.0f} spent "
            "in the previous hour"
        )
    if group == "amount":
        ratio = f["amount_to_avg_ratio"]
        usual = f["amount"] / ratio if ratio > 0 else 0.0
        if ratio >= 1.5:
            return f"£{f['amount']:,.2f} is {ratio:.1f}x this card's average payment"
        if ratio <= 0.3:
            return (
                f"Unusually small payment of £{f['amount']:,.2f} "
                f"(this card usually spends about £{usual:,.0f})"
            )
        return f"£{f['amount']:,.2f}, close to this card's usual £{usual:,.0f}"
    if group == "device":
        if f["is_online"] and f["device_card_count"] >= 1:
            others = int(f["device_card_count"])
            cards = "another card" if others == 1 else f"{others} other cards"
            return f"This device has already been used with {cards}"
        if f["is_online"] and f["is_new_device"]:
            return "Online payment from a device never used with this card before"
        if f["is_online"] and f["new_device_24h"]:
            return "Online payment from a device first seen in the last 24 hours"
        return "Online payment" if f["is_online"] else "In-store payment"
    if group == "timing":
        rarely = ", a time this card is rarely used" if f["hour_familiarity"] < 0.1 else ""
        return f"Made at {f['hour']:02.0f}:00{rarely}"
    if group == "merchant":
        category = str(f["merchant_category"]).replace("_", " ")
        if f["is_new_category"]:
            return f"First ever {category} purchase on this card"
        return f"Merchant category: {category}"
    if group == "history":
        return f"Card has only {_payments(f['n_prev_txns'])} on record"
    raise ValueError(f"Unknown feature group: {group}")


class ReasonExplainer:
    def __init__(self, model, reference: dict, threshold: float, top_k: int = 3) -> None:
        self.model = model
        self.reference = reference
        self.threshold = threshold
        self.top_k = top_k

    @classmethod
    def from_bundle(cls, bundle: dict, features: pd.DataFrame | None = None, top_k: int = 3):
        """Build from a saved model. Older bundles without a reference need ``features``."""
        reference = bundle.get("reference")
        if reference is None:
            if features is None:
                raise ValueError("Bundle has no reference profile; pass training features")
            reference = reference_profile(features)
        return cls(bundle["model"], reference, bundle["threshold"], top_k)

    def shapley(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        """Exact group Shapley values in log-odds.

        Returns (log_odds of each payment (n,), shapley values (n_groups, n),
        log_odds of the reference payment). For every payment:
        log_odds == reference + shapley.sum(axis=0).
        """
        x = rows[FEATURE_COLUMNS].reset_index(drop=True)
        n = len(x)
        # One big table: 128 copies of the batch, coalition by coalition. Built column by
        # column with numpy (fast) rather than 128 DataFrame copies (slow).
        variants = {}
        for g, columns in enumerate(FEATURE_GROUPS.values()):
            keep = np.repeat(MASKS[:, g], n)
            for col in columns:
                real = np.tile(x[col].to_numpy(), len(MASKS))
                variants[col] = np.where(keep, real, self.reference[col])
        table = pd.DataFrame(variants)[FEATURE_COLUMNS]
        values = self.model.decision_function(table).reshape(len(MASKS), n)

        bits = 1 << np.arange(_N_GROUPS)
        mask_ids = MASKS @ bits
        position = np.empty(len(MASKS), dtype=int)
        position[mask_ids] = np.arange(len(MASKS))
        sizes = MASKS.sum(axis=1)

        phi = np.zeros((_N_GROUPS, len(x)))
        for g in range(_N_GROUPS):
            without = np.flatnonzero(~MASKS[:, g])
            with_g = position[mask_ids[without] | bits[g]]
            weights = _WEIGHTS[sizes[without]][:, None]
            phi[g] = (weights * (values[with_g] - values[without])).sum(axis=0)

        full = values[position[mask_ids.max()]]
        empty = float(values[position[0]][0])
        return full, phi, empty

    def explain(self, rows: pd.DataFrame) -> list[dict]:
        """For each payment: score, alert decision and its top reasons, strongest first.

        A reason's ``share`` is the part of this payment's extra suspicion (compared with
        a typical genuine payment) that comes from that group.
        """
        x = rows[FEATURE_COLUMNS].reset_index(drop=True)
        log_odds, phi, _ = self.shapley(x)
        scores = 1 / (1 + np.exp(-log_odds))
        groups = list(FEATURE_GROUPS)
        results = []
        for i in range(len(x)):
            positive = np.clip(phi[:, i], 0, None)
            total = positive.sum()
            reasons = []
            if total > 0:
                for g in np.argsort(-positive)[: self.top_k]:
                    share = positive[g] / total
                    if share >= MIN_SHARE:
                        reasons.append(
                            {
                                "group": groups[g],
                                "share": round(float(share), 3),
                                "text": describe(groups[g], x.iloc[i]),
                            }
                        )
            results.append(
                {
                    "score": float(scores[i]),
                    "flagged": bool(scores[i] >= self.threshold),
                    "reasons": reasons,
                }
            )
        return results
