"""Single source of truth for feature names, constants and shared maths.

Both feature engines (batch for training, online for live scoring) import from here,
so a change to a constant can never make them drift apart.

Golden rule: a transaction's features may only use that card's PAST transactions.
Anything that peeks at the future (or at the label) is data leakage, and makes a model
look great offline and fail in production.
"""

from __future__ import annotations

import numpy as np

# Look-back windows for velocity features: name -> length in seconds.
WINDOWS_S: dict[str, int] = {"10m": 600, "1h": 3_600, "24h": 86_400}
LONGEST_WINDOW_S = max(WINDOWS_S.values())
NEW_DEVICE_WINDOW_S = 86_400  # a device stays "new" for its first 24 hours

SMALL_AMOUNT_GBP = 5.0  # card-testing payments are tiny
NIGHT_END_HOUR = 6  # 00:00-05:59 counts as night
HOUR_BIN_SIZE = 3  # hours are grouped into 8 blocks of 3 hours for habit features
NO_HISTORY_GAP_S = 30 * 86_400  # "seconds since last" for a card's very first payment
MIN_GAP_HOURS = 1 / 60  # floor for travel-speed maths, avoids dividing by zero
ZSCORE_CLIP = 50.0
EARTH_RADIUS_KM = 6_371.0088

# Rough bounding box of the UK: (min_lat, max_lat, min_lon, max_lon).
UK_BOUNDS = (49.8, 60.95, -8.7, 1.8)

NUMERIC_FEATURES = [
    # The payment itself
    "amount",
    "log_amount",
    "is_online",
    "hour",
    "is_night",
    # Velocity: how busy has this card been recently?
    "seconds_since_last",
    "txn_count_10m",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_1h",
    "amount_sum_24h",
    "small_txn_count_10m",
    # Habit: how does this payment compare with the card's own history?
    "n_prev_txns",
    "amount_to_avg_ratio",
    "amount_zscore",
    "hour_familiarity",
    "is_new_category",
    # Location and device
    "dist_from_usual_km",
    "dist_from_last_km",
    "speed_from_last_kmh",
    "is_outside_uk",
    "is_new_device",  # online, and this device has never been used with the card
    "new_device_24h",  # online, from a device first seen on this card less than 24h ago
    # Network: links between cards through shared devices (fraud rings)
    "device_card_count",  # other cards that used this device before (0 in store)
]
CATEGORICAL_FEATURES = ["merchant_category"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ID_COLUMNS = ["transaction_id", "timestamp", "card_id"]
LABEL_COLUMNS = ["is_fraud", "fraud_type", "ring_id"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on plain floats and on numpy/pandas arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def outside_uk(lat, lon):
    """True when a payment happens outside the UK bounding box."""
    min_lat, max_lat, min_lon, max_lon = UK_BOUNDS
    # np.logical_not, not `~`: on a plain Python bool `~True` is -2, which silently broke
    # the online engine (single floats) while the batch engine (arrays) looked fine.
    inside = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
    return np.logical_not(inside)


def amount_stats(n_prev, prev_sum, prev_sum_sq, amount):
    """Ratio to the card's average and z-score, from running sums of past amounts.

    Written once and used by both engines so their arithmetic is identical.
    Returns (amount_to_avg_ratio, amount_zscore).
    """
    n_prev = np.asarray(n_prev, dtype=float)
    amount = np.asarray(amount, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = prev_sum / n_prev
        var = (prev_sum_sq - n_prev * mean * mean) / (n_prev - 1)
        std = np.sqrt(np.clip(var, 0.0, None))
        ratio = np.where(n_prev > 0, amount / mean, 1.0)
        zscore = np.where((n_prev >= 2) & (std > 0), (amount - mean) / std, 0.0)
    return ratio, np.clip(zscore, -ZSCORE_CLIP, ZSCORE_CLIP)
