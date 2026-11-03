"""Live monitor: replay unseen payments through the real-time pipeline, as a bank would."""

from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from fraudlens.app.components import category_name, risk_text, show_reasons, truth_label, where
from fraudlens.app.state import get_world
from fraudlens.app.world import features_frame, strip_labels

SPEEDS = {"Slow": 10, "Normal": 60, "Fast": 300, "Very fast": 1_500}
MAX_LOG = 400


def _reset(world) -> None:
    st.session_state.live = {
        "store": copy.deepcopy(world.store_at_test_start),
        "pos": 0,
        "log": [],
        "alerts": [],
        "processed": 0,
        "frauds_seen": 0,
        "frauds_caught": 0,
        "false_alarms": 0,
        "fraud_amount": 0.0,
        "stopped_amount": 0.0,
    }


def _step(world, n: int) -> None:
    live = st.session_state.live
    batch = world.test_transactions.iloc[live["pos"] : live["pos"] + n]
    if batch.empty:
        return
    txns = batch.to_dict("records")
    features = features_frame([live["store"].process(strip_labels(t)) for t in txns])
    scored = world.score(features)

    for txn, (_, s) in zip(txns, scored.iterrows(), strict=True):
        is_fraud = bool(txn["is_fraud"])
        entry = {
            "time": txn["timestamp"],
            "card": txn["card_id"],
            "amount": txn["amount"],
            "merchant": category_name(txn["merchant_category"]),
            "where": where(txn),
            "risk": s["risk"],
            "alert": bool(s["alert"]),
            "truth": truth_label(txn["is_fraud"], txn["fraud_type"]),
            "reasons": s["reasons"],
            "is_fraud": is_fraud,
        }
        live["log"].append(entry)
        live["processed"] += 1
        if is_fraud:
            live["frauds_seen"] += 1
            live["fraud_amount"] += txn["amount"]
        if entry["alert"]:
            live["alerts"].append(entry)
            if is_fraud:
                live["frauds_caught"] += 1
                live["stopped_amount"] += txn["amount"]
            else:
                live["false_alarms"] += 1
    live["log"] = live["log"][-MAX_LOG:]
    live["pos"] += len(txns)


def _kpis(live) -> None:
    cols = st.columns(5)
    cols[0].metric("Payments scored", f"{live['processed']:,}")
    cols[1].metric("Alerts raised", f"{len(live['alerts']):,}")
    cols[2].metric("Fraud payments caught", f"{live['frauds_caught']} / {live['frauds_seen']}")
    cols[3].metric(
        f"Fraud £ stopped (of £{live['fraud_amount']:,.0f})", f"£{live['stopped_amount']:,.0f}"
    )
    cols[4].metric("False alarms", f"{live['false_alarms']}")


def _stream_table(live) -> None:
    recent = pd.DataFrame(live["log"][-30:][::-1])
    if recent.empty:
        st.info("Press **Run the stream** to start replaying payments.")
        return
    recent["decision"] = recent["alert"].map({True: "ALERT", False: "approved"})
    shown = recent[["time", "amount", "merchant", "where", "risk", "decision"]]

    def highlight(row):
        colour = "background-color: rgba(220, 38, 38, 0.12)" if row["decision"] == "ALERT" else ""
        return [colour] * len(row)

    st.dataframe(
        shown.style.apply(highlight, axis=1),
        hide_index=True,
        width="stretch",
        height=560,
        column_config={
            "time": st.column_config.DatetimeColumn("Time", format="DD MMM HH:mm:ss"),
            "amount": st.column_config.NumberColumn("Amount", format="£%.2f"),
            "merchant": "Merchant",
            "where": "Where",
            "risk": st.column_config.NumberColumn(
                "Risk", format="%.1f", help="Share of genuine payments that look less risky"
            ),
            "decision": "Decision",
        },
    )
    st.caption("Risk 0-100: alerts are raised above 99.9 (riskier than 99.9% of genuine payments).")


def _alert_queue(live) -> None:
    if not live["alerts"]:
        st.caption("No alerts yet.")
        return
    for alert in live["alerts"][-8:][::-1]:
        with st.container(border=True):
            if alert["is_fraud"]:
                st.badge(f"Confirmed {alert['truth'].lower()}", color="red")
            else:
                st.badge("Genuine customer: false alarm", color="gray")
            st.markdown(
                f"**£{alert['amount']:,.2f}** · {alert['merchant']}  \n"
                f"{alert['where']} · {alert['card']} · {alert['time']:%d %b %H:%M}"
            )
            st.caption(risk_text(alert["risk"]))
            show_reasons(alert["reasons"])


def render() -> None:
    world = get_world()
    if "live" not in st.session_state:
        _reset(world)

    st.title("Live monitor")
    st.write(
        "Payments from the test period, which the model has **never seen**, are replayed in "
        "time order. Each one goes through the same path a live payment would: the online "
        "feature engine updates the card's history, the model scores it, and alerts are "
        "explained. The ground truth is shown only so you can judge the decisions."
    )

    controls = st.columns([2, 3, 1])
    controls[0].toggle("Run the stream", key="live_running")
    speed = controls[1].segmented_control(
        "Payments per second", list(SPEEDS), default="Normal", key="live_speed"
    )
    if controls[2].button("Reset", width="stretch"):
        _reset(world)

    running = st.session_state.get("live_running", False)

    @st.fragment(run_every=1.0 if running else None)
    def stream() -> None:
        live = st.session_state.live
        if st.session_state.get("live_running"):
            _step(world, SPEEDS.get(speed or "Normal", 60))
        done = live["pos"] >= len(world.test_transactions)

        clock = (
            live["log"][-1]["time"] if live["log"] else world.test_transactions["timestamp"].iloc[0]
        )
        st.caption(
            f"Simulated clock: **{clock:%a %d %b %Y, %H:%M}** · "
            f"{live['pos']:,} of {len(world.test_transactions):,} test payments replayed"
            + (" · end of the test period reached" if done else "")
        )
        _kpis(live)
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Payment stream")
            _stream_table(live)
        with right:
            st.subheader("Alert queue")
            _alert_queue(live)

    stream()
