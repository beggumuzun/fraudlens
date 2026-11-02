"""Online feature engine: computes features for ONE live payment at a time (for the API).

In production you cannot re-run pandas over the whole history for every payment; you
must answer in milliseconds. So this engine keeps a small running state per card
(running sums, the last 24 hours of payments, devices seen, ...) and updates it after
each payment.

Having two engines creates a classic risk called *training/serving skew*: if they
disagree, the model sees different numbers in production than it was trained on.
``tests/test_features.py`` replays a whole dataset through this engine and checks that
every feature matches the batch engine.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

import pandas as pd

from fraudlens.features.definitions import (
    HOUR_BIN_SIZE,
    LONGEST_WINDOW_S,
    MIN_GAP_HOURS,
    NEW_DEVICE_WINDOW_S,
    NIGHT_END_HOUR,
    NO_HISTORY_GAP_S,
    SMALL_AMOUNT_GBP,
    WINDOWS_S,
    amount_stats,
    haversine_km,
    outside_uk,
)


@dataclass
class CardState:
    """Everything we need to remember about one card."""

    n: int = 0
    sum_amount: float = 0.0
    sum_sq: float = 0.0
    sum_lat: float = 0.0
    sum_lon: float = 0.0
    last_ts: float | None = None  # seconds since epoch
    last_lat: float = 0.0
    last_lon: float = 0.0
    recent: deque = field(default_factory=deque)  # (ts, amount) within the longest window
    devices: dict = field(default_factory=dict)  # device_id -> first time seen
    categories: set = field(default_factory=set)
    hour_bins: Counter = field(default_factory=Counter)


def _seconds(ts) -> float:
    return pd.Timestamp(ts).value / 1e9


def _device(txn: dict) -> str | None:
    device = txn.get("device_id")
    return device if isinstance(device, str) and device else None


class OnlineFeatureStore:
    def __init__(self) -> None:
        self._cards: dict[str, CardState] = {}

    def __len__(self) -> int:
        return len(self._cards)

    def compute(self, txn: dict) -> dict:
        """Features for ``txn`` using only what has been seen so far. Does not change state."""
        state = self._cards.get(txn["card_id"]) or CardState()
        t = _seconds(txn["timestamp"])
        hour = pd.Timestamp(txn["timestamp"]).hour
        amount = float(txn["amount"])
        lat, lon = float(txn["lat"]), float(txn["lon"])
        online = txn["channel"] == "online"
        n = state.n

        features: dict = {
            "amount": amount,
            "log_amount": math.log1p(amount),
            "is_online": float(online),
            "hour": float(hour),
            "is_night": float(hour < NIGHT_END_HOUR),
        }

        # Velocity: payments strictly before now, inside each window [t - w, t).
        gap_s = None if state.last_ts is None else t - state.last_ts
        features["seconds_since_last"] = float(NO_HISTORY_GAP_S if gap_s is None else gap_s)
        past = [(ts, amt) for ts, amt in state.recent if ts < t]
        for name, seconds in WINDOWS_S.items():
            features[f"txn_count_{name}"] = float(sum(1 for ts, _ in past if ts >= t - seconds))
        for name in ("1h", "24h"):
            start = t - WINDOWS_S[name]
            features[f"amount_sum_{name}"] = float(sum(amt for ts, amt in past if ts >= start))
        start_10m = t - WINDOWS_S["10m"]
        features["small_txn_count_10m"] = float(
            sum(1 for ts, amt in past if ts >= start_10m and amt < SMALL_AMOUNT_GBP)
        )

        # Habit
        features["n_prev_txns"] = float(n)
        ratio, zscore = amount_stats(n, state.sum_amount, state.sum_sq, amount)
        features["amount_to_avg_ratio"] = float(ratio)
        features["amount_zscore"] = float(zscore)
        same_bin = state.hour_bins[hour // HOUR_BIN_SIZE]
        features["hour_familiarity"] = same_bin / n if n > 0 else 0.0
        features["is_new_category"] = float(txn["merchant_category"] not in state.categories)

        # Location
        if n > 0:
            usual = haversine_km(state.sum_lat / n, state.sum_lon / n, lat, lon)
            last = haversine_km(state.last_lat, state.last_lon, lat, lon)
            hours = max(gap_s / 3_600, MIN_GAP_HOURS)
            features["dist_from_usual_km"] = float(usual)
            features["dist_from_last_km"] = float(last)
            features["speed_from_last_kmh"] = float(last / hours)
        else:
            features["dist_from_usual_km"] = 0.0
            features["dist_from_last_km"] = 0.0
            features["speed_from_last_kmh"] = 0.0
        features["is_outside_uk"] = float(outside_uk(lat, lon))

        # Device
        device = _device(txn)
        first_seen = state.devices.get(device, t)  # unseen device -> first seen right now
        features["is_new_device"] = float(online and device not in state.devices)
        features["new_device_24h"] = float(online and t - first_seen < NEW_DEVICE_WINDOW_S)

        features["merchant_category"] = txn["merchant_category"]
        return features

    def update(self, txn: dict) -> None:
        """Record ``txn`` in its card's history."""
        state = self._cards.setdefault(txn["card_id"], CardState())
        t = _seconds(txn["timestamp"])
        amount = float(txn["amount"])

        state.n += 1
        state.sum_amount += amount
        state.sum_sq += amount * amount
        state.sum_lat += float(txn["lat"])
        state.sum_lon += float(txn["lon"])
        state.last_ts = t
        state.last_lat, state.last_lon = float(txn["lat"]), float(txn["lon"])
        state.categories.add(txn["merchant_category"])
        state.hour_bins[pd.Timestamp(txn["timestamp"]).hour // HOUR_BIN_SIZE] += 1
        if (device := _device(txn)) is not None:
            state.devices.setdefault(device, t)

        state.recent.append((t, amount))
        while state.recent and state.recent[0][0] < t - LONGEST_WINDOW_S:
            state.recent.popleft()

    def process(self, txn: dict) -> dict:
        """Score-time entry point: compute features, then remember the payment."""
        features = self.compute(txn)
        self.update(txn)
        return features
