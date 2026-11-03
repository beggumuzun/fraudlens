"""Live monitor: replay unseen payments through the real-time pipeline, as a bank would."""

from __future__ import annotations

import copy
import html

import altair as alt
import pandas as pd
import streamlit as st

from fraudlens.app import theme
from fraudlens.app.components import category_name, risk_text, truth_label, where
from fraudlens.app.state import get_world
from fraudlens.app.world import features_frame, strip_labels

SPEEDS = {"Slow": 10, "Normal": 60, "Fast": 300, "Very fast": 1_500}
MAX_LOG = 400  # rows kept for the table and alert queue
TRACE_LENGTH = 3_000  # payments drawn on the seismograph


def _reset(world) -> None:
    st.session_state.live = {
        "store": copy.deepcopy(world.store_at_test_start),
        "pos": 0,
        "log": [],
        "trace": [],
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
            "score": s["score"],
            "risk": s["risk"],
            "alert": bool(s["alert"]),
            "truth": truth_label(txn["is_fraud"], txn["fraud_type"]),
            "reasons": s["reasons"],
            "is_fraud": is_fraud,
        }
        live["processed"] += 1
        live["log"].append(entry)
        live["trace"].append((live["processed"], s["score"], entry["alert"]))
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
    live["trace"] = live["trace"][-TRACE_LENGTH:]
    live["pos"] += len(txns)


def _seismograph(live, threshold: float) -> None:
    """Every payment is a vertical line; the riskier it is, the taller it rises."""
    if not live["trace"]:
        st.markdown(
            '<div class="fl-panel" style="--n:1"><div class="fl-kpi"><div class="fl-kpi-label">'
            "Turn on <b>Run the stream</b> and payments will start drawing here: one line each, "
            "taller the riskier it is. Fraud breaks through the red alert line.</div></div></div>",
            unsafe_allow_html=True,
        )
        return
    trace = pd.DataFrame(live["trace"], columns=["n", "score", "alert"])
    trace["score"] = trace["score"].clip(lower=theme.SCORE_FLOOR)
    trace["level"] = "Ordinary"
    trace.loc[trace["score"] >= threshold / 10, "level"] = "Close to the line"
    trace.loc[trace["alert"], "level"] = "Alert"
    last = int(trace["n"].max())

    x = alt.X("n:Q", axis=None, scale=alt.Scale(domain=[max(last - TRACE_LENGTH, 0), last]))
    y_scale = alt.Scale(type="log", domain=[theme.SCORE_FLOOR, 1])
    levels = alt.Scale(
        domain=["Ordinary", "Close to the line", "Alert"],
        range=[theme.CALM, theme.AMBER, theme.RED],
    )
    base = alt.Chart(trace)
    spikes = base.mark_rule(strokeWidth=1).encode(
        x=x,
        y=alt.Y("score:Q", scale=y_scale, axis=None),
        y2=alt.datum(theme.SCORE_FLOOR),
        color=alt.Color(
            "level:N",
            scale=levels,
            title=None,
            legend=alt.Legend(orient="top", direction="horizontal"),
        ),
    )
    # Every layer reads the same table: Streamlit cannot render a layer chart
    # that mixes several inline datasets.
    heads = (
        base.transform_filter("datum.alert")
        .mark_circle(size=70, color=theme.RED, opacity=1)
        .encode(x=x, y=alt.Y("score:Q", scale=y_scale))
    )
    line = base.mark_rule(strokeDash=[6, 4], color=theme.RED, opacity=0.8).encode(
        y=alt.datum(threshold, scale=y_scale)
    )
    label = (
        base.transform_filter(f"datum.n == {last}")
        .mark_text(align="right", dy=-8, color=theme.RED, fontSize=12)
        .encode(x=x, y=alt.datum(threshold, scale=y_scale), text=alt.value("alert line"))
    )
    theme.show_chart(alt.layer(spikes, line, label, heads).properties(height=230))


def _kpis(live) -> None:
    theme.kpi_panel(
        [
            ("Payments scored", f"{live['processed']:,}", None),
            ("Alerts raised", f"{len(live['alerts']):,}", "alert" if live["alerts"] else None),
            ("Fraud payments caught", f"{live['frauds_caught']} of {live['frauds_seen']}", "ok"),
            (
                f"Stopped, of £{live['fraud_amount']:,.0f} fraud",
                f"£{live['stopped_amount']:,.0f}",
                "ok",
            ),
            ("False alarms", f"{live['false_alarms']}", "warn" if live["false_alarms"] else None),
        ]
    )


def _stream_table(live) -> None:
    recent = pd.DataFrame(live["log"][-30:][::-1])
    if recent.empty:
        st.markdown(
            '<div class="fl-empty">The newest payments will be listed here.</div>',
            unsafe_allow_html=True,
        )
        return
    recent["decision"] = recent["alert"].map({True: "Alert", False: "Approved"})
    shown = recent[["time", "amount", "merchant", "where", "risk", "decision"]]

    def highlight(row):
        colour = "background-color: rgba(255, 77, 94, 0.16)" if row["decision"] == "Alert" else ""
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
    st.caption("Risk runs from 0 to 100. Alerts are raised above 99.9.")


def _alert_queue(live) -> None:
    if not live["alerts"]:
        st.markdown(
            '<div class="fl-empty">No alerts yet. They will queue here with their reasons.</div>',
            unsafe_allow_html=True,
        )
        return
    cards = []
    for alert in live["alerts"][-8:][::-1]:
        false_alarm = not alert["is_fraud"]
        truth = (
            "Genuine customer, false alarm"
            if false_alarm
            else f"Confirmed {alert['truth'].lower()}"
        )
        meta = (
            f"{alert['merchant']}. {alert['where']}. {alert['card']}, "
            f"{alert['time']:%d %b %H:%M}. {risk_text(alert['risk'])}."
        )
        cards.append(
            f'<div class="fl-alert{" fl-alert--false" if false_alarm else ""}">'
            f'<div class="fl-alert-head"><span class="fl-amount">£{alert["amount"]:,.2f}</span>'
            f'<span class="fl-truth">{html.escape(truth)}</span></div>'
            f'<div class="fl-meta">{html.escape(meta)}</div>'
            f"{theme.reasons_html(alert['reasons'])}</div>"
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render() -> None:
    world = get_world()
    if "live" not in st.session_state:
        _reset(world)

    theme.page_header(
        "Live monitor",
        "Payments the model has never seen arrive one by one, in time order. Each goes "
        "through the same path a live payment would: the card's history is updated, the "
        "model scores it, and every alert explains itself. The real outcome is shown only "
        "so you can judge each call.",
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

        _seismograph(live, world.threshold)
        clock = (
            live["log"][-1]["time"] if live["log"] else world.test_transactions["timestamp"].iloc[0]
        )
        st.caption(
            f"Simulated clock {clock:%a %d %b %Y, %H:%M}. "
            f"{live['pos']:,} of {len(world.test_transactions):,} test payments replayed."
            + (" End of the test period." if done else "")
        )
        _kpis(live)
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("Payment stream")
            _stream_table(live)
        with right:
            st.subheader("Alert queue")
            _alert_queue(live)

    stream()
