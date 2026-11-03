"""Online feature engine: computes features for ONE live payment at a time (for the API).

In production you cannot re-run pandas over the whole history for every payment; you
must answer in milliseconds. So this engine keeps a small running state per card
(running sums, the last 24 hours of payments, devices seen, ...) and updates it after
each payment.

Having two engines creates a classic risk called *training/serving skew*: if they
disagree, the model sees different numbers in production than it was trained on.
``tests/test_features.py`` replays a whole dataset through this engine and checks that
every feature matches the batch engine.

Order matters. The batch engine treats payments in (timestamp, transaction_id) order, so
this engine refuses a payment that arrives older than the latest one it has recorded, on
any card (``OutOfOrderError``). The check is global, not per card, because shared devices
link cards: a late payment on card B would also change what card A's device count should
have been. A real system would park such a payment and re-process; silently accepting it
would give negative gaps and features the model never saw in training.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from copy import deepcopy
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
    device_key,
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
    devices: dict = field(default_factory=dict)  # device key -> first time seen
    categories: set = field(default_factory=set)
    hour_bins: Counter = field(default_factory=Counter)


def _seconds(ts) -> float:
    return pd.Timestamp(ts).value / 1e9


def _device(txn: dict) -> str | None:
    device = txn.get("device_id")
    return device if isinstance(device, str) and device else None


def _device_key(txn: dict) -> str:
    return device_key(txn.get("device_id"), txn["channel"] == "online")


class OutOfOrderError(ValueError):
    """A payment arrived older than one already recorded for the same card."""


class OnlineFeatureStore:
    def __init__(self) -> None:
        self._cards: dict[str, CardState] = {}
        # device -> {card: first time seen}: the links between cards (shared devices).
        self._device_cards: dict[str, dict[str, float]] = {}
        self._last: tuple[float, object] | None = None  # (time, transaction_id) of the latest

    def __len__(self) -> int:
        return len(self._cards)

    def snapshot(self, card_ids: list[str]) -> OnlineFeatureStore:
        """An independent copy holding only these cards (cheap 'what if' experiments)."""
        copy_ = OnlineFeatureStore()
        copy_._device_cards = deepcopy(self._device_cards)  # rings span many cards
        copy_._last = self._last
        for card_id in card_ids:
            if card_id in self._cards:
                copy_._cards[card_id] = deepcopy(self._cards[card_id])
        return copy_

    def _check_order(self, txn: dict) -> None:
        if self._last is None:
            return
        t = _seconds(txn["timestamp"])
        txn_id = txn.get("transaction_id")
        last_t, last_id = self._last
        same_time_but_earlier_id = (
            t == last_t and txn_id is not None and last_id is not None and txn_id < last_id
        )
        if t < last_t or same_time_but_earlier_id:
            raise OutOfOrderError(
                f"Payment {txn_id} arrived after {last_id}, but is older: payments must be "
                "processed in (timestamp, transaction_id) order"
            )

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
        key = _device_key(txn)
        first_seen = state.devices.get(key, t)  # unseen device -> first seen right now
        features["is_new_device"] = float(online and key not in state.devices)
        features["new_device_24h"] = float(online and t - first_seen < NEW_DEVICE_WINDOW_S)
        other_cards = self._device_cards.get(device, {}) if device is not None else {}
        features["device_card_count"] = float(
            sum(1 for card, first in other_cards.items() if card != txn["card_id"] and first < t)
        )

        features["merchant_category"] = txn["merchant_category"]
        return features

    def update(self, txn: dict) -> None:
        """Record ``txn`` in its card's history.

        Raises ``OutOfOrderError`` if it is older than the latest payment recorded on any
        card, or has the same timestamp and a smaller transaction_id (the batch tie-break).
        """
        self._check_order(txn)
        state = self._cards.setdefault(txn["card_id"], CardState())
        t = _seconds(txn["timestamp"])
        amount = float(txn["amount"])
        txn_id = txn.get("transaction_id")

        state.n += 1
        state.sum_amount += amount
        state.sum_sq += amount * amount
        state.sum_lat += float(txn["lat"])
        state.sum_lon += float(txn["lon"])
        state.last_ts = t
        self._last = (t, txn_id)
        state.last_lat, state.last_lon = float(txn["lat"]), float(txn["lon"])
        state.categories.add(txn["merchant_category"])
        state.hour_bins[pd.Timestamp(txn["timestamp"]).hour // HOUR_BIN_SIZE] += 1
        state.devices.setdefault(_device_key(txn), t)
        if (device := _device(txn)) is not None:
            self._device_cards.setdefault(device, {}).setdefault(txn["card_id"], t)

        state.recent.append((t, amount))
        while state.recent and state.recent[0][0] < t - LONGEST_WINDOW_S:
            state.recent.popleft()

    def process(self, txn: dict) -> dict:
        """Score-time entry point: compute features, then remember the payment.

        The order check runs first, so a late payment is rejected before it is scored.
        """
        self._check_order(txn)
        features = self.compute(txn)
        self.update(txn)
        return features
