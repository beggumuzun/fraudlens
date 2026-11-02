"""Time-based train / validation / test split.

A random split would let the model train on Tuesday's payments and be tested on Monday's,
which never happens in real life and makes results look better than they are. Here the
model always learns from the past and is judged on the future:

    |------------- train -------------|-- validation --|---- test ----|
    oldest                                                      newest

* train      -> fit the model
* validation -> choose the alert threshold and pick the best model
* test       -> touched once, at the very end, to report honest numbers
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict:
        def part(df: pd.DataFrame) -> dict:
            return {
                "rows": len(df),
                "frauds": int(df["is_fraud"].sum()),
                "from": str(df["timestamp"].min().date()),
                "to": str(df["timestamp"].max().date()),
            }

        return {"train": part(self.train), "valid": part(self.valid), "test": part(self.test)}


def time_split(df: pd.DataFrame, valid_frac: float = 0.15, test_frac: float = 0.2) -> TimeSplit:
    """Split by whole calendar days, so no day is shared between two parts."""
    if not 0 < valid_frac < 1 or not 0 < test_frac < 1 or valid_frac + test_frac >= 1:
        raise ValueError("valid_frac and test_frac must be in (0, 1) and sum to less than 1")

    days = df["timestamp"].dt.floor("D")
    unique_days = days.drop_duplicates().sort_values().tolist()
    n_days = len(unique_days)
    n_test = max(1, round(n_days * test_frac))
    n_valid = max(1, round(n_days * valid_frac))
    if n_test + n_valid >= n_days:
        raise ValueError(f"Not enough days ({n_days}) for this split")

    valid_start = unique_days[n_days - n_test - n_valid]
    test_start = unique_days[n_days - n_test]
    return TimeSplit(
        train=df[days < valid_start],
        valid=df[(days >= valid_start) & (days < test_start)],
        test=df[days >= test_start],
    )
