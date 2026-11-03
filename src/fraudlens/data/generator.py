"""Generate a realistic, labelled stream of card transactions.

Usage (after ``pip install -e .``)::

    fraudlens-generate --customers 2000 --days 30 --out data/transactions.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.data.entities import (
    CATEGORIES,
    ONLINE,
    UK_CITIES,
    Customer,
    MerchantDirectory,
    build_customers,
    build_merchants,
    make_record,
)
from fraudlens.data.scenarios import (
    FRAUD_SCENARIO_WEIGHTS,
    FRAUD_SCENARIOS,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    legit_travel,
)

COLUMNS = [
    "transaction_id",
    "timestamp",
    "customer_id",
    "card_id",
    "merchant_id",
    "merchant_category",
    "merchant_city",
    "channel",
    "amount",
    "lat",
    "lon",
    "device_id",
    "is_fraud",
    "fraud_type",
]
# Columns a live system would never see at scoring time (they are the answer key).
LABEL_COLUMNS = ("is_fraud", "fraud_type")


@dataclass(frozen=True)
class GeneratorConfig:
    n_customers: int = 2_000
    n_days: int = 30
    start_date: str = "2026-01-01"
    seed: int = 42
    fraud_customer_share: float = 0.03  # share of customers hit by one fraud attack
    travel_customer_share: float = 0.03  # share of customers who go abroad (not fraud)
    warmup_days: int = 7  # no fraud at the start, so every card has some history

    def __post_init__(self) -> None:
        if self.n_customers <= 0:
            raise ValueError("n_customers must be positive")
        if self.n_days <= self.warmup_days:
            raise ValueError("n_days must be greater than warmup_days")
        for name in ("fraud_customer_share", "travel_customer_share"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def _normal_transactions(
    customer: Customer, n_days: int, directory: MerchantDirectory, rng: np.random.Generator
) -> list[dict]:
    """Everyday spending: mostly favourite shops, mostly in the home city, mostly daytime."""
    uk_cities = list(UK_CITIES)
    all_categories = list(CATEGORIES)
    records = []
    for _ in range(int(rng.poisson(customer.daily_rate * n_days))):
        day = int(rng.integers(0, n_days))
        if rng.random() < 0.9:
            hour = int(rng.integers(*customer.active_hours))
        else:  # occasional late-night or early-morning purchase
            hour = int(rng.integers(0, 24))
        offset = day * SECONDS_PER_DAY + hour * SECONDS_PER_HOUR
        offset += int(rng.integers(0, SECONDS_PER_HOUR))

        if rng.random() < 0.75:
            category = customer.favourite_categories[
                int(rng.integers(len(customer.favourite_categories)))
            ]
        else:
            category = all_categories[int(rng.integers(len(all_categories)))]
        multiplier, online_share = CATEGORIES[category]

        if rng.random() < online_share:
            merchant = directory.pick(rng, category, ONLINE)
            lat = customer.home_lat + rng.normal(0, 0.01)
            lon = customer.home_lon + rng.normal(0, 0.01)
            device = customer.device_id
        else:
            # 5% of in-store purchases happen while visiting another UK city.
            city = customer.home_city
            if rng.random() < 0.05:
                city = uk_cities[int(rng.integers(len(uk_cities)))]
            merchant = directory.pick(rng, category, city)
            lat, lon, device = merchant.lat, merchant.lon, None

        amount = rng.lognormal(np.log(customer.avg_amount * multiplier), 0.6)
        records.append(
            make_record(
                customer=customer,
                merchant=merchant,
                offset_s=offset,
                amount=amount,
                lat=lat,
                lon=lon,
                device_id=device,
                is_fraud=False,
            )
        )
    return records


def build_world(
    config: GeneratorConfig,
) -> tuple[list[Customer], MerchantDirectory, np.random.Generator]:
    """The customers and merchants of a config, exactly as generate_transactions sees them.

    Used by the app's attack simulator to launch new attacks on the same customers.
    """
    rng = np.random.default_rng(config.seed)
    customers = build_customers(config.n_customers, rng)
    directory = MerchantDirectory(build_merchants(rng))
    return customers, directory, rng


def generate_transactions(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """Build the full labelled dataset, sorted by time. Same config -> identical output."""
    config = config or GeneratorConfig()
    customers, directory, rng = build_world(config)

    records: list[dict] = []
    for customer in customers:
        records.extend(_normal_transactions(customer, config.n_days, directory, rng))

    # Fraud attacks, each on a different victim, only after the warm-up period.
    scenario_names = list(FRAUD_SCENARIOS)
    n_victims = round(config.n_customers * config.fraud_customer_share)
    for idx in rng.choice(config.n_customers, size=n_victims, replace=False):
        name = scenario_names[rng.choice(len(scenario_names), p=FRAUD_SCENARIO_WEIGHTS)]
        day = int(rng.integers(config.warmup_days, config.n_days))
        records.extend(FRAUD_SCENARIOS[name](customers[idx], day, directory, rng))

    # Genuine trips abroad (hard negatives).
    n_travellers = round(config.n_customers * config.travel_customer_share)
    for idx in rng.choice(config.n_customers, size=n_travellers, replace=False):
        day = int(rng.integers(0, config.n_days))
        records.extend(legit_travel(customers[idx], day, directory, rng))

    df = pd.DataFrame.from_records(records)
    # Scenarios that start near the end may run past the last day: cut them off.
    df = df[df["offset_s"] < config.n_days * SECONDS_PER_DAY]
    df = df.sort_values(["offset_s", "customer_id"], kind="stable").reset_index(drop=True)

    df["transaction_id"] = [f"T{i:07d}" for i in range(len(df))]
    df["timestamp"] = pd.Timestamp(config.start_date) + pd.to_timedelta(df["offset_s"], unit="s")
    return df[COLUMNS]


def iter_transactions(df: pd.DataFrame, include_labels: bool = False) -> Iterator[dict]:
    """Replay transactions one at a time in time order, as a live payment system would.

    Labels are dropped by default: a real system only learns the truth days later,
    when the customer reports the fraud, so the scorer must never see them.
    """
    columns = [c for c in df.columns if include_labels or c not in LABEL_COLUMNS]
    for row in df[columns].itertuples(index=False):
        yield row._asdict()


def summarise(df: pd.DataFrame) -> dict:
    """Headline statistics about a generated dataset."""
    return {
        "transactions": len(df),
        "customers": int(df["customer_id"].nunique()),
        "fraud_transactions": int(df["is_fraud"].sum()),
        "fraud_rate_pct": round(100 * float(df["is_fraud"].mean()), 3),
        "fraud_by_type": df["fraud_type"].value_counts().to_dict(),
        "first_timestamp": str(df["timestamp"].min()),
        "last_timestamp": str(df["timestamp"].max()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic card transactions.")
    parser.add_argument("--customers", type=int, default=GeneratorConfig.n_customers)
    parser.add_argument("--days", type=int, default=GeneratorConfig.n_days)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--out", type=Path, default=Path("data/transactions.csv"))
    args = parser.parse_args(argv)

    config = GeneratorConfig(n_customers=args.customers, n_days=args.days, seed=args.seed)
    df = generate_transactions(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Saved {len(df):,} transactions to {args.out}")
    for key, value in summarise(df).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
