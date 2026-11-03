"""Fraud rings: cards linked through shared devices, and the shop where their data leaked."""

from __future__ import annotations

import html
import math

import altair as alt
import pandas as pd
import streamlit as st

from fraudlens.app import theme
from fraudlens.app.components import category_name
from fraudlens.app.state import get_world

COLUMNS = 6
SPACING = 3.6
STATUS_COLOURS = {
    "Flagged card": theme.RED,
    "Linked card, not flagged": theme.AMBER,
    "Family card": theme.TEAL,
    "Shared device": theme.TEXT,
    "Card's own device": theme.CALM,
}


@st.cache_resource(show_spinner=False)
def _links(_world) -> pd.DataFrame:
    tx = _world.transactions
    return tx.loc[tx["device_id"].notna(), ["card_id", "device_id"]].drop_duplicates()


@st.cache_resource(show_spinner=False)
def _constellation(_world) -> pd.DataFrame:
    """One table holding every edge, node and label (a layered chart needs a single dataset)."""
    links = _links(_world)
    cards_per_device = links.groupby("device_id")["card_id"].nunique()
    groups = sorted(_world.groups, key=lambda g: (not g.is_ring, -g.size))
    rows = []
    for i, group in enumerate(groups):
        cx, cy = (i % COLUMNS) * SPACING, -(i // COLUMNS) * SPACING
        # Shared devices (the ring's, or the family laptop) in the middle; the cards around
        # them; each card's own phone just outside its card, so the shared hub stands out.
        mine = links[links["card_id"].isin(group.cards)]
        shared = [d for d in group.devices if cards_per_device.get(d, 0) >= 2]
        own = mine[~mine["device_id"].isin(shared)].groupby("device_id")["card_id"].first()
        pos = {}
        inner = 0.0 if len(shared) == 1 else 0.42
        for k, device in enumerate(shared):
            angle = 2 * math.pi * k / len(shared)
            pos[device] = (cx + inner * math.cos(angle), cy + inner * math.sin(angle))
        card_angle = {}
        for k, card in enumerate(group.cards):
            angle = 2 * math.pi * k / len(group.cards) + math.pi / len(group.cards)
            card_angle[card] = angle
            pos[card] = (cx + 1.05 * math.cos(angle), cy + 1.05 * math.sin(angle))
        for device, card in own.items():
            angle = card_angle[card]
            pos[device] = (cx + 1.5 * math.cos(angle), cy + 1.5 * math.sin(angle))

        for card, device in mine.itertuples(index=False):
            (x, y), (x2, y2) = pos[card], pos[device]
            rows.append(
                {"layer": "edge", "x": x, "y": y, "x2": x2, "y2": y2, "ring": group.is_ring}
            )
        for card in group.cards:
            if not group.is_ring:
                status = "Family card"
            elif card in group.flagged_cards:
                status = "Flagged card"
            else:
                status = "Linked card, not flagged"
            rows.append(
                {
                    "layer": "node",
                    "x": pos[card][0],
                    "y": pos[card][1],
                    "kind": "card",
                    "status": status,
                    "name": card,
                    "group": group.group_id,
                }
            )
        for device in group.devices:
            shared = cards_per_device.get(device, 0) >= 2
            rows.append(
                {
                    "layer": "node",
                    "x": pos[device][0],
                    "y": pos[device][1],
                    "kind": "device",
                    "status": "Shared device" if shared else "Card's own device",
                    "name": device,
                    "group": group.group_id,
                }
            )
        label = f"{group.group_id}  ring" if group.is_ring else f"{group.group_id}  family"
        rows.append(
            {"layer": "label", "x": cx, "y": cy - 1.75, "name": label, "ring": group.is_ring}
        )
    return pd.DataFrame(rows)


def _draw(world) -> None:
    data = _constellation(world)
    n_rows = math.ceil(len(world.groups) / COLUMNS)
    x = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.8, (COLUMNS - 1) * SPACING + 1.8]))
    y = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-(n_rows - 1) * SPACING - 2.3, 1.7]))
    base = alt.Chart(data)
    edges = (
        base.transform_filter("datum.layer == 'edge'")
        .mark_rule(strokeWidth=1.2)
        .encode(
            x=x,
            y=y,
            x2="x2:Q",
            y2="y2:Q",
            color=alt.condition("datum.ring", alt.value(theme.RED), alt.value(theme.CALM)),
            opacity=alt.condition("datum.ring", alt.value(0.55), alt.value(0.7)),
        )
    )
    statuses = list(STATUS_COLOURS)
    nodes = (
        base.transform_filter("datum.layer == 'node'")
        .mark_point(filled=True, opacity=1)
        .encode(
            x=x,
            y=y,
            shape=alt.Shape(
                "kind:N",
                scale=alt.Scale(domain=["card", "device"], range=["circle", "square"]),
                legend=None,
            ),
            size=alt.condition("datum.kind == 'device'", alt.value(110), alt.value(90)),
            color=alt.Color(
                "status:N",
                title=None,
                scale=alt.Scale(domain=statuses, range=[STATUS_COLOURS[s] for s in statuses]),
                legend=alt.Legend(orient="top", columns=5),
            ),
            tooltip=[
                alt.Tooltip("name:N", title="id"),
                alt.Tooltip("status:N", title="role"),
                alt.Tooltip("group:N", title="group"),
            ],
        )
    )
    labels = (
        base.transform_filter("datum.layer == 'label'")
        .mark_text(fontSize=13, fontWeight=600)
        .encode(
            x=x,
            y=y,
            text="name:N",
            color=alt.condition("datum.ring", alt.value(theme.RED), alt.value(theme.MUTED)),
        )
    )
    theme.show_chart(alt.layer(edges, nodes, labels).properties(height=170 * n_rows + 60))
    st.caption(
        "Circles are cards, squares are devices, a line means the card paid from that device. "
        "Hover for ids."
    )


def _ring_detail(world, group) -> None:
    tx = world.transactions
    alerts = world.alerts[world.alerts["card_id"].isin(group.cards)]
    links = _links(world)
    shared = links[links["device_id"].isin(group.devices)].groupby("device_id")["card_id"].nunique()
    flagged_payments = tx.set_index("transaction_id").loc[alerts["transaction_id"]]
    theme.kpi_panel(
        [
            ("Cards linked", f"{group.size}", None),
            ("Cards flagged by the model", f"{len(group.flagged_cards)}", "alert"),
            ("Devices shared across cards", f"{int((shared >= 2).sum())}", "warn"),
            ("Flagged payments", f"£{flagged_payments['amount'].sum():,.0f}", "alert"),
        ]
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Where the card details leaked")
        table = world.compromise.get(group.group_id)
        if table is None or table.empty:
            st.markdown('<div class="fl-empty">No common shop found.</div>', unsafe_allow_html=True)
        else:
            top = table.iloc[0]
            n = len(group.flagged_cards)
            st.markdown(
                f'<p class="fl-lede" style="color:{theme.TEXT}">In the 30 days before their first '
                f"alert, <b>{int(top['victims'])} of {n}</b> flagged cards paid at "
                f"<b>{html.escape(top['merchant_id'])}</b>, an online "
                f"{html.escape(category_name(top['merchant_category']).lower())} shop. Only "
                f"{top['all_cards_share']:.0%} of all customers did: these victims were "
                f"<b>{top['lift']:.1f}x</b> more likely to have shopped there. That shop is "
                "the likely point of compromise.</p>",
                unsafe_allow_html=True,
            )
            if n < 3:
                st.caption(
                    f"Only {n} flagged cards so far, so this is a weak lead: two people "
                    "sharing one shop can be chance. It firms up as more cards are flagged."
                )
            shown = table.assign(
                shop=table["merchant_id"] + ", " + table["merchant_category"].map(category_name),
                victims=table["victims"].astype(int).astype(str) + f" of {n}",
            )[["shop", "victims", "all_cards_share", "lift"]]
            st.dataframe(
                shown,
                hide_index=True,
                width="stretch",
                column_config={
                    "shop": "Shop",
                    "victims": "Victims who paid there",
                    "all_cards_share": st.column_config.NumberColumn(
                        "All customers who did", format="percent"
                    ),
                    "lift": st.column_config.NumberColumn("How unusual", format="%.1fx"),
                },
            )
    with right:
        st.subheader("Cards in the group")
        first = alerts.groupby("card_id")["timestamp"].min()
        cards = pd.DataFrame(
            {
                "card": group.cards,
                "city": [world.cards.loc[c, "usual_city"] for c in group.cards],
                "first_alert": pd.to_datetime([first.get(c, pd.NaT) for c in group.cards]),
            }
        )
        cards["role"] = cards["card"].map(
            lambda c: "Flagged" if c in group.flagged_cards else "Linked only"
        )
        st.dataframe(
            cards.sort_values("first_alert", na_position="last"),
            hide_index=True,
            width="stretch",
            column_config={
                "card": "Card",
                "city": "Lives in",
                "first_alert": st.column_config.DatetimeColumn(
                    "First alert", format="DD MMM HH:mm"
                ),
                "role": "Role",
            },
        )
        st.caption(
            "Linked but never flagged usually means a family member of a victim who shares "
            "their device: linked by the graph, cleared by the model."
        )


def render() -> None:
    world = get_world()
    theme.page_header(
        "Fraud rings",
        "One criminal, many cards. Cards are linked when they pay from the same device. A "
        "linked group is called a ring only when the model has flagged at least two of its "
        "cards, so families sharing a laptop are not accused. Then the victims' own history "
        "points to the shop where their card details leaked. Only alerts from after the "
        "model's training period are used: it cannot simply remember the answers.",
    )
    rings = [g for g in world.groups if g.is_ring]
    theme.kpi_panel(
        [
            ("Linked groups checked", f"{len(world.groups)}", None),
            ("Rings found", f"{len(rings)}", "alert"),
            ("Cards in rings", f"{sum(g.size for g in rings)}", "warn"),
            ("Families cleared", f"{len(world.groups) - len(rings)}", "ok"),
        ]
    )
    _draw(world)

    if not rings:
        return
    st.subheader("Investigate a ring")
    group = st.selectbox(
        "Ring",
        rings,
        format_func=lambda g: f"{g.group_id}: {g.size} cards, {len(g.flagged_cards)} flagged",
        key="ring_pick",
    )
    _ring_detail(world, group)
