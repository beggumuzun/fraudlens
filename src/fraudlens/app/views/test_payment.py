"""Test a payment: invent a payment for a real card and see the model's decision and why."""

from __future__ import annotations

from datetime import datetime, time, timedelta

import pandas as pd
import streamlit as st

from fraudlens.app import theme
from fraudlens.app.components import (
    ALL_CITIES,
    card_picker,
    card_profile,
    category_name,
    risk_text,
    show_reasons,
)
from fraudlens.app.state import get_world
from fraudlens.app.world import features_frame
from fraudlens.data.entities import CATEGORIES, UK_CITIES
from fraudlens.explain.reasons import FEATURE_GROUPS
from fraudlens.features.definitions import FEATURE_COLUMNS

NEW_DEVICE = "NEW-DEVICE"


def _presets(card) -> dict:
    """Ready-made scenarios, tailored to the selected card."""
    avg = float(card["avg_amount"])
    home = card["usual_city"]
    other_city = "Manchester" if home == "London" else "London"
    next_day = (pd.Timestamp(card["last_payment"]) + timedelta(days=1)).date()
    usual_hour = int(min(max(card["usual_hour"], 8), 21))
    return {
        "Everyday purchase": dict(
            amount=round(avg, 2),
            channel="In store",
            category="grocery",
            city=home,
            device="usual",
            date=next_day,
            time=time(usual_hour, 15),
        ),
        "Night shopping abroad": dict(
            amount=round(max(avg * 12, 400.0), 2),
            channel="Online",
            category="jewellery",
            city="Lagos",
            device="new",
            date=next_day,
            time=time(3, 10),
        ),
        "Big spend in another city": dict(
            amount=round(avg * 5, 2),
            channel="In store",
            category="electronics",
            city=other_city,
            device="usual",
            date=next_day,
            time=time(15, 30),
        ),
        "Dinner on holiday": dict(
            amount=round(avg * 1.2, 2),
            channel="In store",
            category="restaurant",
            city="Barcelona",
            device="usual",
            date=next_day,
            time=time(20, 30),
        ),
    }


def _apply_preset(values: dict) -> None:
    for key, value in values.items():
        st.session_state[f"tp_{key}"] = value


def render() -> None:
    world = get_world()
    theme.page_header(
        "Test a payment",
        "Pick a real customer from the demo bank, describe a new payment and see what the "
        "model decides. Everything is worked out from that card's own history, exactly as "
        "it would be for a live payment.",
    )

    card_id = card_picker(world, key="tp_card")
    card = world.cards.loc[card_id]
    card_profile(world, card_id)

    presets = _presets(card)
    if st.session_state.get("tp_for_card") != card_id:  # new card: load its everyday preset
        _apply_preset(presets["Everyday purchase"])
        st.session_state["tp_for_card"] = card_id

    st.markdown("**Try a scenario**")
    preset_cols = st.columns(len(presets))
    for col, (name, values) in zip(preset_cols, presets.items(), strict=True):
        col.button(name, on_click=_apply_preset, args=(values,), width="stretch")

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("The payment")
        amount = st.number_input("Amount (£)", min_value=0.5, step=5.0, key="tp_amount")
        channel = st.radio("Channel", ["In store", "Online"], horizontal=True, key="tp_channel")
        category = st.selectbox(
            "Merchant category", list(CATEGORIES), format_func=category_name, key="tp_category"
        )
        uk = list(UK_CITIES)
        city = st.selectbox(
            "Where the shop is" if channel == "In store" else "Where the buyer's device appears",
            list(ALL_CITIES),
            format_func=lambda c: c if c in uk else f"{c} (abroad)",
            key="tp_city",
        )
        device = "usual"
        if channel == "Online":
            device = st.radio(
                "Device",
                ["usual", "new"],
                format_func=lambda d: (
                    "The customer's usual device" if d == "usual" else "A new, unknown device"
                ),
                horizontal=True,
                key="tp_device",
            )
        date_col, time_col = st.columns(2)
        day = date_col.date_input("Date", key="tp_date")
        clock = time_col.time_input("Time", key="tp_time", step=timedelta(minutes=5))

    timestamp = pd.Timestamp(datetime.combine(day, clock))
    last = pd.Timestamp(card["last_payment"])
    if timestamp <= last:
        st.warning(
            f"The card's last payment was at {last:%d %b %H:%M}; the test payment must come "
            "after it. Using one minute later instead."
        )
        timestamp = last + timedelta(minutes=1)

    online = channel == "Online"
    lat, lon = ALL_CITIES[city]
    txn = {
        "transaction_id": "TEST",
        "timestamp": timestamp,
        "card_id": card_id,
        "merchant_category": category,
        "channel": "online" if online else "in_store",
        "amount": float(amount),
        "lat": lat,
        "lon": lon,
        "device_id": (card["usual_device"] if device == "usual" else NEW_DEVICE)
        if online
        else None,
    }
    features = features_frame([world.store_full.compute(txn)])  # compute() never changes state
    scored = world.score(features, explain=False).iloc[0]
    reasons = world.explain_one(features)

    with right:
        st.subheader("The decision")
        if scored["alert"]:
            theme.decision(
                True,
                "Held for review",
                f"Risk {scored['risk']:.2f} out of 100. {risk_text(scored['risk'])}, so the "
                "payment is paused and sent to a fraud analyst.",
            )
        else:
            theme.decision(
                False,
                "Approved",
                f"Risk {scored['risk']:.2f} out of 100. {risk_text(scored['risk'])}; alerts "
                "start at 99.9.",
            )
        theme.risk_meter(float(scored["score"]), world.threshold)
        st.markdown("**What pushed the risk up**")
        show_reasons(reasons, "This payment looks like the card's normal behaviour.")

        with st.expander(f"All {len(FEATURE_COLUMNS)} features the model saw"):
            values = features.iloc[0]
            rows = [
                {"group": group, "feature": col, "value": values[col]}
                for group, cols in FEATURE_GROUPS.items()
                for col in cols
            ]
            table = pd.DataFrame(rows)
            table["value"] = table["value"].map(
                lambda v: f"{v:,.2f}" if isinstance(v, float) else str(v)
            )
            st.dataframe(table, hide_index=True, width="stretch")
