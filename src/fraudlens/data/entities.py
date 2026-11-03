"""Building blocks of the synthetic world: cities, merchant categories, customers, merchants.

Everything random goes through a single ``numpy.random.Generator`` passed in by the caller,
so the same seed always produces exactly the same world.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# Approximate city-centre coordinates (lat, lon) where customers live.
UK_CITIES: dict[str, tuple[float, float]] = {
    "Bristol": (51.4545, -2.5879),
    "London": (51.5072, -0.1276),
    "Manchester": (53.4808, -2.2426),
    "Birmingham": (52.4862, -1.8904),
    "Leeds": (53.8008, -1.5491),
    "Glasgow": (55.8642, -4.2518),
    "Cardiff": (51.4816, -3.1791),
    "Edinburgh": (55.9533, -3.1883),
}
# Share of the customer population living in each UK city (same order as UK_CITIES).
UK_CITY_WEIGHTS = np.array([0.25, 0.25, 0.10, 0.10, 0.08, 0.08, 0.07, 0.07])

# Cities abroad, used both by fraudsters and by genuine customers on holiday.
FOREIGN_CITIES: dict[str, tuple[float, float]] = {
    "Lagos": (6.5244, 3.3792),
    "Bucharest": (44.4268, 26.1025),
    "Sao Paulo": (-23.5505, -46.6333),
    "Manila": (14.5995, 120.9842),
    "Barcelona": (41.3874, 2.1686),
    "Antalya": (36.8969, 30.7133),
}

# category -> (amount multiplier relative to customer's average spend, share of online sales)
CATEGORIES: dict[str, tuple[float, float]] = {
    "grocery": (0.6, 0.15),
    "restaurant": (0.8, 0.05),
    "transport": (0.4, 0.30),
    "fuel": (1.0, 0.00),
    "clothing": (1.5, 0.50),
    "entertainment": (0.9, 0.60),
    "health": (1.0, 0.20),
    "electronics": (4.0, 0.60),
    "jewellery": (6.0, 0.30),
    "digital_goods": (0.3, 1.00),
}
HIGH_RISK_CATEGORIES = ("electronics", "jewellery", "digital_goods")
EVERYDAY_CATEGORIES = tuple(c for c in CATEGORIES if c not in ("electronics", "jewellery"))

ONLINE = "ONLINE"  # pseudo-city used for online merchants


@dataclass(frozen=True)
class Customer:
    customer_id: str
    card_id: str
    home_city: str
    home_lat: float
    home_lon: float
    avg_amount: float  # typical spend in GBP
    daily_rate: float  # expected number of transactions per day
    active_hours: tuple[int, int]  # [start, end) hour of usual activity
    favourite_categories: tuple[str, ...]
    device_id: str  # the customer's own phone/laptop used for online payments


@dataclass(frozen=True)
class Merchant:
    merchant_id: str
    category: str
    city: str  # a real city, or ONLINE
    lat: float  # NaN for online merchants
    lon: float
    online: bool


def build_customers(n: int, rng: np.random.Generator) -> list[Customer]:
    """Create ``n`` customers, each with their own spending habits."""
    city_names = list(UK_CITIES)
    customers: list[Customer] = []
    for i in range(n):
        city = city_names[rng.choice(len(city_names), p=UK_CITY_WEIGHTS)]
        lat, lon = UK_CITIES[city]
        favourites = rng.choice(len(EVERYDAY_CATEGORIES), size=3, replace=False)
        customers.append(
            Customer(
                customer_id=f"C{i:05d}",
                card_id=f"CARD{i:05d}",
                home_city=city,
                home_lat=float(lat + rng.normal(0, 0.03)),
                home_lon=float(lon + rng.normal(0, 0.03)),
                # Log-normal: most people spend ~£25 per purchase, a few spend much more.
                avg_amount=round(float(rng.lognormal(mean=np.log(25), sigma=0.5)), 2),
                daily_rate=float(rng.uniform(0.8, 3.5)),
                active_hours=(int(rng.integers(6, 11)), int(rng.integers(20, 24))),
                favourite_categories=tuple(EVERYDAY_CATEGORIES[j] for j in favourites),
                device_id=f"D{i:05d}",
            )
        )
    return customers


def build_merchants(
    rng: np.random.Generator,
    stores_per_city_category: int = 3,
    online_per_category: int = 15,
) -> list[Merchant]:
    """Create physical stores in every city plus online shops for each category."""
    all_cities = {**UK_CITIES, **FOREIGN_CITIES}
    merchants: list[Merchant] = []

    def next_id() -> str:
        return f"M{len(merchants):05d}"

    for category, (_, online_share) in CATEGORIES.items():
        if online_share < 1.0:  # category has physical stores
            for city, (lat, lon) in all_cities.items():
                for _ in range(stores_per_city_category):
                    merchants.append(
                        Merchant(
                            merchant_id=next_id(),
                            category=category,
                            city=city,
                            lat=float(lat + rng.normal(0, 0.02)),
                            lon=float(lon + rng.normal(0, 0.02)),
                            online=False,
                        )
                    )
        if online_share > 0.0:  # category sells online
            for _ in range(online_per_category):
                merchants.append(
                    Merchant(next_id(), category, ONLINE, float("nan"), float("nan"), True)
                )
    return merchants


class MerchantDirectory:
    """Fast lookup of merchants by (category, city)."""

    def __init__(self, merchants: list[Merchant]) -> None:
        self._by_key: dict[tuple[str, str], list[Merchant]] = defaultdict(list)
        for merchant in merchants:
            self._by_key[(merchant.category, merchant.city)].append(merchant)

    def pick(self, rng: np.random.Generator, category: str, city: str) -> Merchant:
        options = self._by_key.get((category, city))
        if not options:
            raise KeyError(f"No merchant for category={category!r}, city={city!r}")
        return options[int(rng.integers(len(options)))]


def make_record(
    *,
    customer: Customer,
    merchant: Merchant,
    offset_s: int,
    amount: float,
    lat: float,
    lon: float,
    device_id: str | None,
    is_fraud: bool,
    fraud_type: str | None = None,
    ring_id: str | None = None,
    breach_ring_id: str | None = None,
) -> dict:
    """One transaction as a plain dict. ``offset_s`` = seconds since the simulation start.

    Ground truth, never a model input: ``ring_id`` names the fraud ring behind an attack,
    ``breach_ring_id`` marks the genuine payment where a ring stole the card's details.
    """
    return {
        "offset_s": int(offset_s),
        "customer_id": customer.customer_id,
        "card_id": customer.card_id,
        "merchant_id": merchant.merchant_id,
        "merchant_category": merchant.category,
        "merchant_city": merchant.city,
        "channel": "online" if merchant.online else "in_store",
        "amount": round(max(float(amount), 0.5), 2),
        "lat": round(float(lat), 4),
        "lon": round(float(lon), 4),
        "device_id": device_id,
        "is_fraud": int(is_fraud),
        "fraud_type": fraud_type,
        "ring_id": ring_id,
        "breach_ring_id": breach_ring_id,
    }
