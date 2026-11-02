"""Behaviour patterns injected on top of normal spending.

Fraud scenarios are modelled on well-known real-world attack types:

* card_testing      - a fraudster checks a stolen card number with many tiny online
                      payments, then makes a big purchase once the card is confirmed to work.
* account_takeover  - someone gains access to the customer's online account and buys
                      high-value goods at night, from a new device in another country.
* stolen_card_spree - the physical card is stolen and used in shops in another UK city
                      for several large purchases within a few hours.

``legit_travel`` is the opposite: a genuine customer on holiday abroad. It is NOT fraud,
but it looks suspicious (foreign location), so the model has to learn the difference.
These "hard negatives" stop the model from simply flagging every foreign transaction.

Every scenario has the same signature: (customer, day, directory, rng) -> list of records.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fraudlens.data.entities import (
    CATEGORIES,
    FOREIGN_CITIES,
    HIGH_RISK_CATEGORIES,
    ONLINE,
    UK_CITIES,
    Customer,
    MerchantDirectory,
    make_record,
)

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400

Scenario = Callable[[Customer, int, MerchantDirectory, np.random.Generator], list[dict]]


def _random_foreign_city(rng: np.random.Generator) -> tuple[str, float, float]:
    names = list(FOREIGN_CITIES)
    name = names[int(rng.integers(len(names)))]
    lat, lon = FOREIGN_CITIES[name]
    return name, lat + rng.normal(0, 0.03), lon + rng.normal(0, 0.03)


def _fraudster_device(rng: np.random.Generator) -> str:
    return f"FD{int(rng.integers(1_000_000)):06d}"


def card_testing(
    customer: Customer, day: int, directory: MerchantDirectory, rng: np.random.Generator
) -> list[dict]:
    device = _fraudster_device(rng)
    _, lat, lon = _random_foreign_city(rng)
    t = day * SECONDS_PER_DAY + int(rng.integers(0, 24)) * SECONDS_PER_HOUR
    records = []

    # Phase 1: a burst of tiny payments, seconds apart.
    for _ in range(int(rng.integers(5, 13))):
        t += int(rng.integers(10, 90))
        merchant = directory.pick(rng, "digital_goods", ONLINE)
        records.append(
            make_record(
                customer=customer,
                merchant=merchant,
                offset_s=t,
                amount=rng.uniform(0.5, 3.0),
                lat=lat,
                lon=lon,
                device_id=device,
                is_fraud=True,
                fraud_type="card_testing",
            )
        )

    # Phase 2: card works, so cash out with expensive electronics.
    for _ in range(int(rng.integers(1, 3))):
        t += int(rng.integers(120, 1_800))
        merchant = directory.pick(rng, "electronics", ONLINE)
        records.append(
            make_record(
                customer=customer,
                merchant=merchant,
                offset_s=t,
                amount=rng.uniform(300, 1_500),
                lat=lat,
                lon=lon,
                device_id=device,
                is_fraud=True,
                fraud_type="card_testing",
            )
        )
    return records


def account_takeover(
    customer: Customer, day: int, directory: MerchantDirectory, rng: np.random.Generator
) -> list[dict]:
    device = _fraudster_device(rng)
    _, lat, lon = _random_foreign_city(rng)
    # Happens in the middle of the night (01:00-05:00) while the real customer is asleep.
    t = day * SECONDS_PER_DAY + int(rng.integers(1, 5)) * SECONDS_PER_HOUR
    t += int(rng.integers(0, SECONDS_PER_HOUR))
    records = []
    for _ in range(int(rng.integers(2, 5))):
        category = HIGH_RISK_CATEGORIES[int(rng.integers(len(HIGH_RISK_CATEGORIES)))]
        merchant = directory.pick(rng, category, ONLINE)
        records.append(
            make_record(
                customer=customer,
                merchant=merchant,
                offset_s=t,
                amount=customer.avg_amount * rng.uniform(5, 15),
                lat=lat,
                lon=lon,
                device_id=device,
                is_fraud=True,
                fraud_type="account_takeover",
            )
        )
        t += int(rng.integers(300, 3_600))
    return records


def stolen_card_spree(
    customer: Customer, day: int, directory: MerchantDirectory, rng: np.random.Generator
) -> list[dict]:
    other_cities = [c for c in UK_CITIES if c != customer.home_city]
    city = other_cities[int(rng.integers(len(other_cities)))]
    categories = ("electronics", "clothing", "jewellery", "fuel", "grocery")
    t = day * SECONDS_PER_DAY + int(rng.integers(10, 18)) * SECONDS_PER_HOUR
    records = []
    for _ in range(int(rng.integers(4, 9))):
        category = categories[int(rng.integers(len(categories)))]
        merchant = directory.pick(rng, category, city)
        records.append(
            make_record(
                customer=customer,
                merchant=merchant,
                offset_s=t,
                amount=customer.avg_amount * rng.uniform(2, 6),
                lat=merchant.lat,
                lon=merchant.lon,
                device_id=None,
                is_fraud=True,
                fraud_type="stolen_card_spree",
            )
        )
        t += int(rng.integers(600, 2_700))  # 10-45 minutes between shops
    return records


def legit_travel(
    customer: Customer, day: int, directory: MerchantDirectory, rng: np.random.Generator
) -> list[dict]:
    """A genuine holiday abroad: normal amounts and hours, but a foreign location."""
    city, _, _ = _random_foreign_city(rng)
    holiday_categories = ("restaurant", "grocery", "transport", "entertainment", "clothing")
    records = []
    for trip_day in range(day, day + int(rng.integers(3, 8))):
        for _ in range(int(rng.poisson(customer.daily_rate))):
            hour = int(rng.integers(*customer.active_hours))
            t = trip_day * SECONDS_PER_DAY + hour * SECONDS_PER_HOUR
            t += int(rng.integers(0, SECONDS_PER_HOUR))
            category = holiday_categories[int(rng.integers(len(holiday_categories)))]
            merchant = directory.pick(rng, category, city)
            multiplier, _ = CATEGORIES[category]
            amount = rng.lognormal(np.log(customer.avg_amount * multiplier), 0.6)
            records.append(
                make_record(
                    customer=customer,
                    merchant=merchant,
                    offset_s=t,
                    amount=amount,
                    lat=merchant.lat,
                    lon=merchant.lon,
                    device_id=None,
                    is_fraud=False,
                )
            )
    return records


FRAUD_SCENARIOS: dict[str, Scenario] = {
    "card_testing": card_testing,
    "account_takeover": account_takeover,
    "stolen_card_spree": stolen_card_spree,
}
FRAUD_SCENARIO_WEIGHTS = np.array([0.35, 0.35, 0.30])
