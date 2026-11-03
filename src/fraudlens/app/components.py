"""Small UI building blocks shared by the pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from fraudlens.app import theme
from fraudlens.data.entities import FOREIGN_CITIES, UK_CITIES
from fraudlens.features.definitions import haversine_km

ALL_CITIES = {**UK_CITIES, **FOREIGN_CITIES}

ATTACK_LABELS = {
    "card_testing": "Card testing",
    "account_takeover": "Account takeover",
    "stolen_card_spree": "Stolen card spree",
}


def nearest_city(lat: float, lon: float) -> str:
    """Name of the closest known city, to show where an online payment appears to come from."""
    names = list(ALL_CITIES)
    coords = np.array([ALL_CITIES[n] for n in names])
    distances = haversine_km(coords[:, 0], coords[:, 1], lat, lon)
    return names[int(np.argmin(distances))]


def where(row) -> str:
    if row["channel"] == "online":
        return f"Online, from {nearest_city(row['lat'], row['lon'])}"
    return f"In store, {row['merchant_city']}"


def category_name(category: str) -> str:
    return category.replace("_", " ").capitalize()


def truth_label(is_fraud: int, fraud_type) -> str:
    if is_fraud:
        return f"Fraud: {ATTACK_LABELS.get(fraud_type, fraud_type)}"
    return "Genuine"


def show_reasons(reasons: list[dict], empty_text: str = "Nothing unusual stands out.") -> None:
    """Each reason with a bar showing its share of the suspicion."""
    st.markdown(theme.reasons_html(reasons, empty_text), unsafe_allow_html=True)


def risk_text(risk: float) -> str:
    if risk >= 99.995:
        return "Riskier than every genuine payment in the validation weeks"
    return f"Riskier than {risk:.2f}% of genuine payments"


def card_picker(world, key: str, label: str = "Card") -> str:
    """Searchable card selector showing each card's home city and typical spend."""
    cards = world.cards
    genuine_busy = cards[(cards["frauds"] == 0) & (cards["payments"] >= 40)].index
    options = list(cards.index)
    default = options.index(genuine_busy[0]) if len(genuine_busy) else 0

    def fmt(card_id: str) -> str:
        c = cards.loc[card_id]
        return f"{card_id}, {c['usual_city']}, £{c['avg_amount']:.0f} average"

    return st.selectbox(label, options, index=default, format_func=fmt, key=key)


def card_profile(world, card_id: str, per_row: int = 4) -> None:
    """The card's normal behaviour as an instrument strip (per_row=2 for narrow columns)."""
    c = world.cards.loc[card_id]
    tiles = [
        ("Usual city", c["usual_city"], None),
        ("Average payment", f"£{c['avg_amount']:.2f}", None),
        ("Payments on record", f"{c['payments']:,}", None),
        ("Last payment", pd.Timestamp(c["last_payment"]).strftime("%d %b, %H:%M"), None),
    ]
    for start in range(0, len(tiles), per_row):
        theme.kpi_panel(tiles[start : start + per_row])
