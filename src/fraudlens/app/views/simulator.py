"""Attack simulator: launch a fresh attack on a real card and watch when it gets caught."""

from __future__ import annotations

import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

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
        st.error(f"**Missed.** No payment raised an alert: £{total:,.2f} lost.")
        return
    first = int(alerts[0])
    lost = result.loc[: first - 1, "amount"].sum()
    minutes = (result.loc[first, "timestamp"] - result.loc[0, "timestamp"]).total_seconds() / 60
    st.success(
        f"**Caught on payment {first + 1} of {len(result)}**, "
        f"{minutes:.0f} minutes into the attack. If the card is blocked at the first alert, "
        f"**£{total - lost:,.2f} of £{total:,.2f} is stopped** (£{lost:,.2f} got through)."
    )


def _chart(result: pd.DataFrame, threshold: float) -> None:
    """Raw model score on a log scale.

    The 0-100 risk percentile saturates near 100 for every attack payment (99.8 vs 99.99
    look identical), while the raw score spreads them out by orders of magnitude.
    """
    data = result[["step", "amount", "risk", "score", "alert"]].assign(
        status=np.where(result["alert"], "Alert", "Passed"), threshold=threshold
    )
    low = min(float(data["score"].min()), threshold) / 3
    x = alt.X("step:O", title="Payment in the attack", axis=alt.Axis(labelAngle=0))
    y_scale = alt.Scale(type="log", domain=[low, 1])
    base = alt.Chart(data)
    line = base.mark_line(color="#94a3b8").encode(
        x=x, y=alt.Y("score:Q", title="Fraud score (log scale)", scale=y_scale)
    )
    points = base.mark_circle(size=140, opacity=1).encode(
        x=x,
        y=alt.Y("score:Q", scale=y_scale),
        color=alt.Color(
            "status:N",
            title=None,
            scale=alt.Scale(domain=["Alert", "Passed"], range=["#dc2626", "#64748b"]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            "step",
            alt.Tooltip("amount:Q", title="amount (£)", format=",.2f"),
            alt.Tooltip("risk:Q", format=".2f"),
        ],
    )
    rule = base.mark_rule(strokeDash=[4, 4], color="#dc2626").encode(
        y=alt.Y("threshold:Q", scale=y_scale)
    )
    st.altair_chart(alt.layer(line, rule, points).properties(height=280), width="stretch")
    st.caption(
        "Dashed line: the alert threshold. Each step up the axis is 10 times more suspicious."
    )


def _steps(result: pd.DataFrame, animate: bool) -> None:
    placeholder = st.empty()
    first_alert = int(result.index[result["alert"]][0]) if result["alert"].any() else None
    for upto in range(1, len(result) + 1) if animate else [len(result)]:
        with placeholder.container():
            for i in range(upto):
                row = result.iloc[i]
                if row["alert"]:
                    status = "ALERT"
                elif first_alert is not None and i > first_alert:
                    status = "Would be blocked"
                else:
                    status = "Passed"
                with st.container(border=True):
                    cols = st.columns([1, 3, 2, 2])
                    cols[0].markdown(f"**#{int(row['step'])}**  \n{row['timestamp']:%H:%M:%S}")
                    cols[1].markdown(
                        f"**£{row['amount']:,.2f}** · {category_name(row['merchant_category'])}  \n"
                        f"{where(row)}"
                    )
                    cols[2].markdown(f"Risk **{row['risk']:.2f}**")
                    color = {"ALERT": "red", "Would be blocked": "orange", "Passed": "gray"}[status]
                    cols[3].badge(status, color=color)
                    with st.expander("Why", expanded=(i == first_alert)):
                        show_reasons(row["reasons"])
        if animate:
            time.sleep(0.5)


def render() -> None:
    world = get_world()
    st.title("Attack simulator")
    st.write(
        "Launch a brand-new attack on a real customer's card. The attack is generated on the "
        "spot, after the end of the data, so the model has never seen it. Each payment is "
        "scored in order, on top of the card's genuine history."
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
        st.info("Choose a card and an attack, then press **Launch attack**.")
        return

    result = _run_attack(world, *key)
    st.divider()
    _summary(result)
    _chart(result, world.threshold)
    _steps(result, animate=st.session_state.pop("sim_animate", False))
