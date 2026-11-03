"""Model performance: baseline vs gradient boosting, and the false-alarm trade-off."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from fraudlens.app import theme
from fraudlens.app.components import ATTACK_LABELS
from fraudlens.app.state import get_world
from fraudlens.explain.cli import explain_test_alerts
from fraudlens.model.metrics import evaluate, threshold_at_fpr

BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]  # false alarms per 1,000 genuine payments
MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}


@st.cache_resource(show_spinner=False)
def _tradeoff(_world) -> pd.DataFrame:
    """Test-set results of the saved model for every false-alarm budget."""
    rows = []
    test_days = _world.test_scored["timestamp"].dt.normalize().nunique()
    for budget in BUDGETS:
        threshold = threshold_at_fpr(
            _world.valid_scored["is_fraud"], _world.valid_scored["score"], budget / 1_000
        )
        m = evaluate(_world.test_scored, threshold)
        rows.append(
            {
                "budget": budget,
                "fraud caught": m["recall"],
                "£ caught": m["value_recall"],
                "attacks caught": m["attack_recall"],
                "precision": m["precision"],
                "alerts per day": m["alerts"] / test_days,
                "by type": m["recall_by_type"],
                "attacks": m["attacks"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def _reason_table(_world) -> pd.DataFrame:
    table, _ = explain_test_alerts(_world.features, _world.bundle)
    return table


def render() -> None:
    world = get_world()
    bundle = world.bundle
    split = bundle["split"]

    theme.page_header(
        "Model performance",
        f"Trained on {split['train']['from']} to {split['train']['to']}, alert line set on "
        f"{split['valid']['from']} to {split['valid']['to']}, and tested on "
        f"{split['test']['from']} to {split['test']['to']}: {split['test']['rows']:,} payments "
        f"the model never saw, {split['test']['frauds']} of them fraudulent.",
    )

    st.subheader("Baseline vs gradient boosting")
    comparison = pd.DataFrame(
        {
            MODEL_NAMES[name]: {
                "PR-AUC": c["test"]["average_precision"],
                "Fraud payments caught": c["test"]["recall"],
                "£ fraud caught": c["test"]["value_recall"],
                "Attacks caught": c["test"]["attack_recall"],
                "Alert precision": c["test"]["precision"],
            }
            for name, c in bundle["candidates"].items()
        }
    ).T
    st.dataframe(
        comparison.style.format("{:.1%}", subset=comparison.columns[1:]).format(
            "{:.3f}", subset=["PR-AUC"]
        ),
        width="stretch",
    )
    served = MODEL_NAMES[bundle["model_name"]]
    if bundle.get("selected_by") == "fixed":
        choice = (
            f"The app serves **{served}**, the clear winner of the full 60-day evaluation "
            "(651,379 payments, 29 attacks; see the README). This demo world is much smaller, "
            "with only a handful of attacks to test on, so the two models can swap places "
            "here: that is noise, and exactly why the choice is made on the larger evaluation."
        )
    else:
        choice = (
            f"The saved model is **{served}**, picked on validation data, never on the test set."
        )
    st.caption(
        "Both models use the same alert budget: at most 1 false alarm per 1,000 genuine "
        f"payments. {choice}"
    )

    st.subheader("The false-alarm trade-off")
    st.write(
        "Every bank chooses how many genuine customers it is willing to disturb. Move the "
        "slider to see what a stricter or looser budget means for the saved model."
    )
    tradeoff = _tradeoff(world)
    budget = st.select_slider(
        "False alarms allowed per 1,000 genuine payments", options=BUDGETS, value=1.0
    )
    row = tradeoff[tradeoff["budget"] == budget].iloc[0]
    attacks_caught = round(row["attacks caught"] * row["attacks"])
    theme.kpi_panel(
        [
            ("Fraud payments caught", f"{row['fraud caught']:.1%}", "ok"),
            ("Fraud money caught", f"{row['£ caught']:.1%}", "ok"),
            ("Attacks caught", f"{attacks_caught} of {row['attacks']}", "ok"),
            ("Alerts per day", f"{row['alerts per day']:.0f}", "warn"),
            ("Alerts that are real fraud", f"{row['precision']:.0%}", None),
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        curve = tradeoff.melt(
            id_vars=["budget", "alerts per day"],
            value_vars=["fraud caught", "£ caught"],
            var_name="measure",
            value_name="share",
        )
        chart = (
            alt.Chart(curve)
            .mark_line(point=True)
            .encode(
                x=alt.X("alerts per day:Q", title="Alerts per day"),
                y=alt.Y("share:Q", title="Share caught", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "measure:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["fraud caught", "£ caught"], range=[theme.AMBER, theme.TEAL]
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=["budget", alt.Tooltip("share:Q", format=".1%"), "alerts per day"],
            )
            .properties(height=300)
        )
        theme.show_chart(chart)
        st.caption("More alerts catch more fraud, but each extra alert costs analyst time.")
    with right:
        by_type = pd.DataFrame(
            {
                "attack": [ATTACK_LABELS[k] for k in row["by type"]],
                "caught": list(row["by type"].values()),
            }
        )
        chart = (
            alt.Chart(by_type)
            .mark_bar(color=theme.TEAL, cornerRadiusEnd=2)
            .encode(
                x=alt.X(
                    "caught:Q",
                    title="Fraud payments caught",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y("attack:N", title=None, sort="-x"),
                tooltip=["attack", alt.Tooltip("caught:Q", format=".1%")],
            )
            .properties(height=180)
        )
        theme.show_chart(chart)
        st.caption("Stolen-card sprees are the hardest: they look like a genuine day out.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("What the model relies on")
        importance = (
            pd.Series(bundle["feature_importance"])
            .head(10)
            .rename_axis("feature")
            .reset_index(name="importance")
        )
        chart = (
            alt.Chart(importance)
            .mark_bar(color=theme.AMBER, cornerRadiusEnd=2)
            .encode(
                x=alt.X("importance:Q", title="Drop in PR-AUC when shuffled"),
                y=alt.Y("feature:N", title=None, sort="-x", axis=alt.Axis(labelLimit=220)),
            )
            .properties(height=300)
        )
        theme.show_chart(chart)
    with right:
        st.subheader("Why alerts were raised")
        st.write("Top reason behind each test-period alert, by what it really was:")
        st.dataframe(_reason_table(world), width="stretch")
