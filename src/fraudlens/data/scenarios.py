"""Behaviour patterns injected on top of normal spending.

Fraud scenarios are modelled on well-known real-world attack types:

* card_testing      - a fraudster checks a stolen card number with many tiny online
                      payments, then makes a big purchase once the card is confirmed to work.
                      Half of them appear to be in the UK (VPN / proxy).
* account_takeover  - someone gains access to the customer's online account and buys
                      high-value goods at night from a new device, usually from abroad
                      (30% appear to be in the UK).
* stolen_card_spree - the physical card is stolen and used in shops in another UK city
                      for several large purchases within a few hours.

Online attacks are carried out by an ``Actor``: the device(s) and apparent location of the
criminal. A lone fraudster attacks one card. A **fraud ring** is one actor attacking many
cards from the same small pool of devices, after buying the card details stolen from one
breached shop (the "point of compromise"), which every victim had paid at a few days before.

``legit_travel`` is the opposite: a genuine customer on holiday abroad. It is NOT fraud,
but it looks suspicious (foreign location), so the model has to learn the difference.
These "hard negatives" stop the model from simply flagging every foreign transaction.

Every scenario has the signature (customer, day, directory, rng, actor=None) -> records.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fraudlens.data.entities import (
    CATEGORIES,
    FOREIGN_CITIES,
    HIGH_RISK_CATEGORIES,
    ONLINE,
    UK_CITIES,
    Customer,
    Merchant,
    MerchantDirectory,
    make_record,
)

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400

# Online shops whose customer data leaks in a breach (everyday shops many people use).
BREACHABLE_CATEGORIES = ("clothing", "entertainment", "grocery", "health")


@dataclass(frozen=True)
class Actor:
    """Who is behind an online attack: their device pool and where they appear to be."""

    devices: tuple[str, ...]
    lat: float
    lon: float
    ring_id: str | None = None  # None for a lone fraudster

    def device(self, rng: np.random.Generator) -> str:
        return self.devices[int(rng.integers(len(self.devices)))]


@dataclass(frozen=True)
class Ring:
    """A fraud ring: one actor, plus the breached shop its card details came from."""

    actor: Actor
    breached_merchant: Merchant


Scenario = Callable[..., list[dict]]


def _random_foreign_city(rng: np.random.Generator) -> tuple[str, float, float]:
    names = list(FOREIGN_CITIES)
    name = names[int(rng.integers(len(names)))]
    lat, lon = FOREIGN_CITIES[name]
    return name, lat + rng.normal(0, 0.03), lon + rng.normal(0, 0.03)


def _fraudster_location(rng: np.random.Generator, uk_share: float) -> tuple[float, float]:
    """Where an online fraudster appears to be.

    Many hide behind a UK VPN or proxy, so they look like they are in a UK city. Without
    this, a model can learn the shortcut "online fraud always comes from abroad" and miss
    every domestic attack (this exact shortcut was found by the Phase 4 explanations).
    """
    if rng.random() < uk_share:
        names = list(UK_CITIES)
        lat, lon = UK_CITIES[names[int(rng.integers(len(names)))]]
        return lat + rng.normal(0, 0.03), lon + rng.normal(0, 0.03)
    _, lat, lon = _random_foreign_city(rng)
    return lat, lon


def _fraudster_device(rng: np.random.Generator) -> str:
    return f"FD{int(rng.integers(1_000_000)):06d}"


def lone_actor(rng: np.random.Generator, uk_share: float) -> Actor:
    lat, lon = _fraudster_location(rng, uk_share)
    return Actor(devices=(_fraudster_device(rng),), lat=lat, lon=lon)


def make_ring(ring_id: str, directory: MerchantDirectory, rng: np.random.Generator) -> Ring:
    """A ring works from 1-3 devices in one apparent location, with one breached shop."""
    lat, lon = _fraudster_location(rng, uk_share=0.4)
    devices = tuple(_fraudster_device(rng) for _ in range(int(rng.integers(1, 4))))
    category = BREACHABLE_CATEGORIES[int(rng.integers(len(BREACHABLE_CATEGORIES)))]
    return Ring(
        actor=Actor(devices=devices, lat=lat, lon=lon, ring_id=ring_id),
        breached_merchant=directory.pick(rng, category, ONLINE),
    )


def breach_purchase(
    customer: Customer, attack_day: int, ring: Ring, rng: np.random.Generator
) -> dict:
    """The genuine payment, days before the attack, where the victim's card data leaked."""
    day = max(attack_day - int(rng.integers(2, 11)), 0)
    hour = int(rng.integers(*customer.active_hours))
    t = day * SECONDS_PER_DAY + hour * SECONDS_PER_HOUR + int(rng.integers(0, SECONDS_PER_HOUR))
    multiplier, _ = CATEGORIES[ring.breached_merchant.category]
    return make_record(
        customer=customer,
        merchant=ring.breached_merchant,
        offset_s=t,
        amount=rng.lognormal(np.log(customer.avg_amount * multiplier), 0.5),
        lat=customer.home_lat + rng.normal(0, 0.01),
        lon=customer.home_lon + rng.normal(0, 0.01),
        device_id=customer.device_id,
        is_fraud=False,
    )


def card_testing(
    customer: Customer,
    day: int,
    directory: MerchantDirectory,
    rng: np.random.Generator,
    actor: Actor | None = None,
) -> list[dict]:
    actor = actor or lone_actor(rng, uk_share=0.5)
    t = day * SECONDS_PER_DAY + int(rng.integers(0, 24)) * SECONDS_PER_HOUR
    records = []

    def add(merchant: Merchant, amount: float) -> None:
        records.append(
            make_record(
                customer=customer,
                merchant=merchant,
                offset_s=t,
                amount=amount,
                lat=actor.lat,
                lon=actor.lon,
                device_id=actor.device(rng),  # rings rotate through their devices
                is_fraud=True,
                fraud_type="card_testing",
                ring_id=actor.ring_id,
            )
        )

    # Phase 1: a burst of tiny payments, seconds apart.
    for _ in range(int(rng.integers(5, 13))):
        t += int(rng.integers(10, 90))
        add(directory.pick(rng, "digital_goods", ONLINE), rng.uniform(0.5, 3.0))

    # Phase 2: card works, so cash out with expensive electronics.
    for _ in range(int(rng.integers(1, 3))):
        t += int(rng.integers(120, 1_800))
        add(directory.pick(rng, "electronics", ONLINE), rng.uniform(300, 1_500))
    return records


def account_takeover(
    customer: Customer,
    day: int,
    directory: MerchantDirectory,
    rng: np.random.Generator,
    actor: Actor | None = None,
) -> list[dict]:
    actor = actor or lone_actor(rng, uk_share=0.3)
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
                lat=actor.lat,
                lon=actor.lon,
                device_id=actor.device(rng),
                is_fraud=True,
                fraud_type="account_takeover",
                ring_id=actor.ring_id,
            )
        )
        t += int(rng.integers(300, 3_600))
    return records


def stolen_card_spree(
    customer: Customer,
    day: int,
    directory: MerchantDirectory,
    rng: np.random.Generator,
    actor: Actor | None = None,  # unused: a stolen physical card is used in shops
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
ONLINE_SCENARIOS = ("card_testing", "account_takeover")  # the ones rings can carry out
