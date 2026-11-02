"""Batch feature engine: computes every feature for a whole dataset at once (for training).

It is vectorised with pandas, so 130k transactions take a few seconds. Every aggregate
only looks at the same card's earlier transactions; ``tests/test_features.py`` proves it.

Usage (after ``pip install -e .``)::

    fraudlens-features --input data/transactions.csv --out data/features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.features.definitions import (
    FEATURE_COLUMNS,
    HOUR_BIN_SIZE,
    ID_COLUMNS,
    LABEL_COLUMNS,
    MIN_GAP_HOURS,
    NEW_DEVICE_WINDOW_S,
    NIGHT_END_HOUR,
    NO_HISTORY_GAP_S,
    NUMERIC_FEATURES,
    SMALL_AMOUNT_GBP,
    WINDOWS_S,
    amount_stats,
    haversine_km,
    outside_uk,
)

IN_STORE_DEVICE = "__in_store__"


def _prev_cumsum(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Running total of ``values`` per card, EXCLUDING the current row."""
    return values.groupby(groups, sort=False).cumsum().groupby(groups, sort=False).shift(1)


def _rolling_prev(df: pd.DataFrame, column: str, seconds: int, how: str) -> np.ndarray:
    """Count or sum of ``column`` over the card's payments in [t - window, t).

    ``closed="left"`` excludes the current payment (and anything at the exact same
    moment), so the window only ever contains the past.
    """
    rolled = (
        df.set_index("timestamp")
        .groupby("card_id", sort=True)[column]
        .rolling(f"{seconds}s", closed="left")
    )
    result = rolled.count() if how == "count" else rolled.sum()
    return result.fillna(0.0).to_numpy()


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return one row of features per transaction, in the same order as the input."""
    df = transactions.reset_index(drop=True)
    # Process card by card, oldest first. The index keeps each row's original position.
    df = df.loc[df.sort_values(["card_id", "timestamp", "transaction_id"], kind="stable").index]
    card = df["card_id"]
    by_card = df.groupby("card_id", sort=False)

    out = df[ID_COLUMNS].copy()
    amount = df["amount"]
    n_prev = by_card.cumcount()

    # --- The payment itself ---
    out["amount"] = amount
    out["log_amount"] = np.log1p(amount)
    out["is_online"] = (df["channel"] == "online").astype(int)
    hour = df["timestamp"].dt.hour
    out["hour"] = hour
    out["is_night"] = (hour < NIGHT_END_HOUR).astype(int)

    # --- Velocity ---
    prev_ts = by_card["timestamp"].shift(1)
    gap_s = (df["timestamp"] - prev_ts).dt.total_seconds()
    out["seconds_since_last"] = gap_s.fillna(NO_HISTORY_GAP_S)

    rolling_input = df[["card_id", "timestamp", "amount"]].assign(
        small=(amount < SMALL_AMOUNT_GBP).astype(float)
    )
    for name, seconds in WINDOWS_S.items():
        out[f"txn_count_{name}"] = _rolling_prev(rolling_input, "amount", seconds, "count")
    out["amount_sum_1h"] = _rolling_prev(rolling_input, "amount", WINDOWS_S["1h"], "sum")
    out["amount_sum_24h"] = _rolling_prev(rolling_input, "amount", WINDOWS_S["24h"], "sum")
    out["small_txn_count_10m"] = _rolling_prev(rolling_input, "small", WINDOWS_S["10m"], "sum")

    # --- Habit ---
    out["n_prev_txns"] = n_prev
    prev_sum = _prev_cumsum(amount, card).fillna(0.0)
    prev_sum_sq = _prev_cumsum(amount * amount, card).fillna(0.0)
    ratio, zscore = amount_stats(n_prev, prev_sum, prev_sum_sq, amount)
    out["amount_to_avg_ratio"] = ratio
    out["amount_zscore"] = zscore

    hour_bin = hour // HOUR_BIN_SIZE
    same_bin_prev = df.assign(_bin=hour_bin).groupby(["card_id", "_bin"]).cumcount()
    out["hour_familiarity"] = np.where(n_prev > 0, same_bin_prev / n_prev.clip(lower=1), 0.0)

    category_first_use = df.groupby(["card_id", "merchant_category"]).cumcount() == 0
    out["is_new_category"] = category_first_use.astype(int)

    # --- Location ---
    has_history = n_prev > 0
    usual_lat = _prev_cumsum(df["lat"], card) / n_prev.where(has_history)
    usual_lon = _prev_cumsum(df["lon"], card) / n_prev.where(has_history)
    dist_usual = haversine_km(usual_lat, usual_lon, df["lat"], df["lon"])
    out["dist_from_usual_km"] = dist_usual.fillna(0.0)

    dist_last = haversine_km(
        by_card["lat"].shift(1), by_card["lon"].shift(1), df["lat"], df["lon"]
    ).fillna(0.0)
    out["dist_from_last_km"] = dist_last
    hours_since_last = (gap_s / 3_600).clip(lower=MIN_GAP_HOURS)
    out["speed_from_last_kmh"] = (dist_last / hours_since_last).fillna(0.0)
    out["is_outside_uk"] = outside_uk(df["lat"], df["lon"]).astype(int)

    # --- Device ---
    online = df["channel"] == "online"
    by_device = df.assign(_device=df["device_id"].fillna(IN_STORE_DEVICE)).groupby(
        ["card_id", "_device"]
    )
    out["is_new_device"] = (online & (by_device.cumcount() == 0)).astype(int)
    # cummin = earliest time this device was seen with this card, up to and including now.
    device_age_s = (df["timestamp"] - by_device["timestamp"].cummin()).dt.total_seconds()
    out["new_device_24h"] = (online & (device_age_s < NEW_DEVICE_WINDOW_S)).astype(int)

    out["merchant_category"] = df["merchant_category"]
    for label in LABEL_COLUMNS:
        if label in df.columns:
            out[label] = df[label]

    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].astype(float)
    labels = [c for c in LABEL_COLUMNS if c in out.columns]
    return out.sort_index()[ID_COLUMNS + FEATURE_COLUMNS + labels]


def feature_report(features: pd.DataFrame) -> pd.DataFrame:
    """Average of each numeric feature for genuine vs fraudulent payments."""
    means = features.groupby("is_fraud")[NUMERIC_FEATURES].mean().T
    means.columns = ["genuine_mean", "fraud_mean"]
    means["fraud_to_genuine"] = means["fraud_mean"] / means["genuine_mean"].replace(0, np.nan)
    return means.round(3)


def load_transactions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build point-in-time fraud features.")
    parser.add_argument("--input", type=Path, default=Path("data/transactions.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/features.csv"))
    args = parser.parse_args(argv)

    features = build_features(load_transactions(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)

    print(f"Saved {len(features):,} rows x {len(FEATURE_COLUMNS)} features to {args.out}\n")
    if "is_fraud" in features.columns:
        with pd.option_context("display.width", 120):
            print(feature_report(features).to_string())


if __name__ == "__main__":
    main()
