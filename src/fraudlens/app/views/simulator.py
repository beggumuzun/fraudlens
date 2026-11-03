"""Attack simulator: launch a fresh attack on a real card and watch when it gets caught."""

from __future__ import annotations

import html
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from fraudlens.app import theme
from fraudlens.app.components import (
    ATTACK_LABELS,
    card_picker,
    card_profile,
    category_name,
    show_reasons,
    where,
)
from fraudlens.app.state import get_world
from fraudlens.app.world import features_frame
from fraudlens.data.scenarios import FRAUD_SCENARIOS

DESCRIPTIONS = {
    "card_testing": (
        "A criminal has the card number. They try a burst of tiny online payments to check "
        "it works, then buy expensive electronics. Half of them hide behind a UK VPN."
    ),
    "account_takeover": (
        "Someone has the customer's online login. In the middle of the night, from a new "
        "device, they buy high-value goods. Usually abroad, sometimes through a UK VPN."
    ),
    "stolen_card_spree": (
        "The physical card was stolen. The thief goes shopping in another UK city, several "
        "big purchases within a few hours, before the card is reported."
    ),
}


def _run_attack(world, card_id: str, attack: str, seed: int) -> pd.DataFrame:
    """Generate the attack, then score it payment by payment on top of the card's history."""
    rng = np.random.default_rng(seed)
    customer = world.customer(card_id)
    last_day = (
        pd.Timestamp(world.cards.loc[card_id, "last_payment"]).normalize()
        - pd.Timestamp(world.config.start_date)
    ).days
    records = FRAUD_SCENARIOS[attack](customer, last_day + 1, world.directory, rng)
    records.sort(key=lambda r: r["offset_s"])

    store = world.store_full.snapshot([card_id])  # private copy: the real history is untouched
    start = pd.Timestamp(world.config.start_date)
    txns, feature_rows = [], []
    for i, record in enumerate(records, start=1):
        txn = {k: v for k, v in record.items() if k not in ("offset_s", "is_fraud", "fraud_type")}
        txn["transaction_id"] = f"SIM{i:03d}"
        txn["timestamp"] = start + pd.Timedelta(seconds=record["offset_s"])
        feature_rows.append(store.process(txn))
        txns.append(txn)

    features = features_frame(feature_rows)
    scored = world.score(features, explain=False)
    explanations = world.explainer.explain(features)
    out = pd.DataFrame(txns)
    out["step"] = range(1, len(out) + 1)
    out["score"] = scored["score"].to_numpy()
    out["risk"] = scored["risk"].to_numpy()
    out["alert"] = scored["alert"].to_numpy()
    out["reasons"] = [e["reasons"] for e in explanations]
    return out


def _summary(result: pd.DataFrame) -> None:
    total = result["amount"].sum()
    alerts = result.index[result["alert"]]
    if len(alerts) == 0:
        theme.decision(
            True, "Missed", f"No payment in this attack crossed the alert line: £{total:,.2f} lost."
        )
        return
    first = int(alerts[0])
    lost = result.loc[: first - 1, "amount"].sum()
    minutes = (result.loc[first, "timestamp"] - result.loc[0, "timestamp"]).total_seconds() / 60
    if first == 0:
        title = "Caught on the very first payment"
        sub = f"With the card blocked straight away, <b>all £{total:,.2f} is stopped</b>."
    else:
        title = f"Caught on payment {first + 1} of {len(result)}"
        sub = (
            f"{minutes:.0f} minutes into the attack. With the card blocked at that first alert, "
            f"<b>£{total - lost:,.2f} of £{total:,.2f} is stopped</b> and £{lost:,.2f} gets "
            "through."
        )
    theme.decision(False, title, sub)


def _chart(result: pd.DataFrame, threshold: float) -> None:
    """Raw model score on a log scale.

    The 0-100 risk percentile saturates near 100 for every attack payment (99.8 vs 99.99
    look identical), while the raw score spreads them out by orders of magnitude.
    """
    data = result[["step", "amount", "risk", "score", "alert"]].assign(
        status=np.where(result["alert"], "Alert", "Passed"), threshold=threshold
    )
    data["score"] = data["score"].clip(lower=theme.SCORE_FLOOR)
    low = max(min(float(data["score"].min()), threshold) / 3, theme.SCORE_FLOOR)
    x = alt.X("step:O", title="Payment in the attack", axis=alt.Axis(labelAngle=0))
    y_scale = alt.Scale(type="log", domain=[low, 1])
    decades = [10.0**k for k in range(-6, 1) if 10.0**k >= low]
    y_axis = alt.Axis(values=decades, format="~g", gridOpacity=0.6)
    base = alt.Chart(data)
    line = base.mark_line(color=theme.MUTED, strokeWidth=1.5).encode(
        x=x, y=alt.Y("score:Q", title="Fraud score, log scale", scale=y_scale, axis=y_axis)
    )
    points = base.mark_circle(size=160, opacity=1).encode(
        x=x,
        y=alt.Y("score:Q", scale=y_scale),
        color=alt.Color(
            "status:N",
            title=None,
            scale=alt.Scale(domain=["Alert", "Passed"], range=[theme.RED, theme.CALM]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            "step",
            alt.Tooltip("amount:Q", title="amount (£)", format=",.2f"),
            alt.Tooltip("risk:Q", format=".2f"),
        ],
    )
    rule = base.mark_rule(strokeDash=[6, 4], color=theme.RED, opacity=0.8).encode(
        y=alt.Y("threshold:Q", scale=y_scale)
    )
    theme.show_chart(alt.layer(line, rule, points).properties(height=280))
    st.caption(
        "The dashed line is the alert line. Each step up the axis is ten times more suspicious."
    )


STATUS = {
    "alert": ("Alert raised", "fl-step--alert", "fl-status--alert"),
    "blocked": ("Card already blocked", "fl-step--blocked", "fl-status--blocked"),
    "passed": ("Passed", "", "fl-status--passed"),
}


def _step_html(row, status: str) -> str:
    label, row_class, status_class = STATUS[status]
    return (
        f'<div class="fl-step {row_class}">'
        f'<div><div class="fl-step-n">{int(row["step"])}</div>'
        f'<div class="fl-step-time">{row["timestamp"]:%H:%M:%S}</div></div>'
        f'<div class="fl-step-what">£{row["amount"]:,.2f} on '
        f"{html.escape(category_name(row['merchant_category']).lower())}<br>"
        f"<span>{html.escape(where(row))}</span></div>"
        f'<div class="fl-step-risk">{row["risk"]:.2f}</div>'
        f'<div class="fl-status {status_class}">{label}</div></div>'
    )


def _steps(result: pd.DataFrame, animate: bool) -> None:
    st.subheader("Payment by payment")
    placeholder = st.empty()
    first_alert = int(result.index[result["alert"]][0]) if result["alert"].any() else None
    for upto in range(1, len(result) + 1) if animate else [len(result)]:
        with placeholder.container():
            for i in range(upto):
                row = result.iloc[i]
                if row["alert"] and (first_alert is None or i <= first_alert):
                    status = "alert"
                elif first_alert is not None and i > first_alert:
                    status = "blocked"
                else:
                    status = "passed"
                st.markdown(_step_html(row, status), unsafe_allow_html=True)
                with st.expander("Why this score", expanded=(i == first_alert)):
                    show_reasons(row["reasons"])
        if animate:
            time.sleep(0.5)


def render() -> None:
    world = get_world()
    theme.page_header(
        "Attack simulator",
        "Launch a brand-new attack on a real customer's card. It is generated on the spot, "
        "after the end of the data, so the model has never seen it. Each payment is scored "
        "in order on top of the card's genuine history.",
    )

    left, right = st.columns([2, 3], gap="large")
    with left:
        card_id = card_picker(world, key="sim_card", label="Victim's card")
        attack = st.radio(
            "Attack type", list(ATTACK_LABELS), format_func=ATTACK_LABELS.get, key="sim_attack"
        )
        st.caption(DESCRIPTIONS[attack])
        animate = st.toggle("Play payments one by one", value=True)
        if st.button("Launch attack", type="primary", width="stretch"):
            st.session_state.sim_seed = int(np.random.default_rng().integers(1_000_000))
            st.session_state.sim_key = (card_id, attack, st.session_state.sim_seed)
            st.session_state.sim_animate = animate
    with right:
        st.markdown("**The victim's normal behaviour**")
        card_profile(world, card_id, per_row=2)

    key = st.session_state.get("sim_key")
    if not key or key[:2] != (card_id, attack):
        st.markdown(
            '<div class="fl-empty">Choose a card and an attack, then press '
            "<b>Launch attack</b>.</div>",
            unsafe_allow_html=True,
        )
        return

    result = _run_attack(world, *key)
    st.divider()
    _summary(result)
    _chart(result, world.threshold)
    _steps(result, animate=st.session_state.pop("sim_animate", False))
