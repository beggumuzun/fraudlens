"""Fraud-ring analysis: link cards through shared devices, then find where their data leaked.

Step 1, linking. Every online payment links a card to a device. Union-find joins these
links into groups: cards connected directly or through a chain of shared devices.

Step 2, separating rings from families. Structure alone is not enough: a family of three
sharing a laptop forms a group of three cards too. A group is called a *ring* only when
at least ``min_flagged`` of its cards have been flagged by the fraud model. Graph links
and model alerts each catch what the other misses.

Step 3, the point of compromise. A ring buys stolen card details, usually from one
breached shop. So: which merchant did the ring's victims pay at, in the weeks before they
were attacked, far more often than customers in general? That shop is the likely leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fraudlens.graph.union_find import UnionFind


@dataclass
class LinkedGroup:
    """Cards joined through shared devices."""

    group_id: str
    cards: list[str]
    devices: list[str]
    flagged_cards: list[str] = field(default_factory=list)
    is_ring: bool = False

    @property
    def size(self) -> int:
        return len(self.cards)


def linked_groups(transactions: pd.DataFrame, min_cards: int = 2) -> list[LinkedGroup]:
    """Groups of at least ``min_cards`` cards connected through the devices they paid from."""
    online = transactions.loc[transactions["device_id"].notna(), ["card_id", "device_id"]]
    links = online.drop_duplicates()
    uf = UnionFind()
    for card, device in links.itertuples(index=False):
        uf.union(("card", card), ("device", device))

    groups = []
    for members in uf.groups():
        cards = sorted(m[1] for m in members if m[0] == "card")
        if len(cards) >= min_cards:
            devices = sorted(m[1] for m in members if m[0] == "device")
            groups.append(LinkedGroup(group_id="", cards=cards, devices=devices))
    groups.sort(key=lambda g: (-g.size, g.cards[0]))
    for i, group in enumerate(groups, start=1):
        group.group_id = f"G{i:03d}"
    return groups


def find_rings(
    transactions: pd.DataFrame, flagged_cards: set[str], min_cards: int = 3, min_flagged: int = 2
) -> list[LinkedGroup]:
    """Linked groups of at least ``min_cards`` cards, marked as rings when enough are flagged."""
    groups = linked_groups(transactions, min_cards=min_cards)
    for group in groups:
        group.flagged_cards = [c for c in group.cards if c in flagged_cards]
        group.is_ring = len(group.flagged_cards) >= min_flagged
    return groups


def point_of_compromise(
    transactions: pd.DataFrame,
    first_alert: dict[str, pd.Timestamp],
    lookback_days: int = 30,
    top: int = 5,
) -> pd.DataFrame:
    """Merchants the victims paid at before their first alert, ranked by how unusual that is.

    ``first_alert`` maps each victim card to the time of its first alert. For each victim
    we take its payments in the ``lookback_days`` before that moment: the leak happened
    before the attack. No payment is filtered out, so fraud payments can be among them:
    ones the model missed, or, when only recent alerts are passed in (out-of-sample
    analysis), earlier ones it did flag. They rarely matter, because fraudsters' shops
    differ from victim to victim while the breached shop is shared by all of them.

    ``lift`` compares the share of victims who paid at a shop with the share of all cards
    that paid there over the same period, which ends at the ring's latest first alert, so
    nothing from after the investigation point is used. A breached shop scores far above 1.
    """
    columns = ["merchant_id", "merchant_category", "merchant_city", "victims", "lift"]
    if not first_alert:
        return pd.DataFrame(columns=columns)
    ts = transactions
    victims = list(first_alert)
    window_start = min(first_alert.values()) - pd.Timedelta(days=lookback_days)
    window_end = max(first_alert.values())
    before = []
    for card in victims:
        end = first_alert[card]
        start = end - pd.Timedelta(days=lookback_days)
        in_window = (ts["timestamp"] >= start) & (ts["timestamp"] < end)
        rows = ts[(ts["card_id"] == card) & in_window]
        before.append(rows[["card_id", "merchant_id", "merchant_category", "merchant_city"]])
    visited = pd.concat(before).drop_duplicates(["card_id", "merchant_id"])

    per_merchant = visited.groupby(["merchant_id", "merchant_category", "merchant_city"]).agg(
        victims=("card_id", "nunique")
    )
    period = ts[(ts["timestamp"] >= window_start) & (ts["timestamp"] < window_end)]
    reach = period.groupby("merchant_id")["card_id"].nunique() / period["card_id"].nunique()
    table = per_merchant.reset_index()
    table["victim_share"] = table["victims"] / len(victims)
    table["all_cards_share"] = table["merchant_id"].map(reach)
    table["lift"] = table["victim_share"] / table["all_cards_share"]
    table = table.sort_values(["victims", "lift"], ascending=False)
    return table.head(top).reset_index(drop=True)


def ring_report(
    transactions: pd.DataFrame, alerts: pd.DataFrame, min_cards: int = 3, min_flagged: int = 2
) -> tuple[list[LinkedGroup], dict[str, pd.DataFrame]]:
    """Rings plus the top point-of-compromise candidates for each.

    ``alerts`` needs columns card_id and timestamp: the payments the model flagged.
    """
    first_alert = alerts.groupby("card_id")["timestamp"].min().to_dict()
    groups = find_rings(transactions, set(first_alert), min_cards, min_flagged)
    compromise = {
        g.group_id: point_of_compromise(transactions, {c: first_alert[c] for c in g.flagged_cards})
        for g in groups
        if g.is_ring
    }
    return groups, compromise


def _true_rings(transactions: pd.DataFrame) -> dict[str, set[str]]:
    rings = transactions.dropna(subset=["ring_id"])
    return rings.groupby("ring_id")["card_id"].agg(set).to_dict()


def true_breached_shops(transactions: pd.DataFrame) -> dict[str, str]:
    """Answer key: ring id -> the shop where its victims' card details leaked."""
    if "breach_ring_id" not in transactions:
        return {}
    breaches = transactions.dropna(subset=["breach_ring_id"])
    return breaches.groupby("breach_ring_id")["merchant_id"].first().to_dict()


def score_against_truth(
    groups: list[LinkedGroup],
    transactions: pd.DataFrame,
    compromise: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """How well the detected rings match the real ones (ground truth labels, for evaluation).

    A real ring counts as found when a detected ring overlaps it with Jaccard >= 0.5. With
    ``compromise`` given, also counts the found rings whose top-ranked shop is the shop that
    was really breached.
    """
    truth = _true_rings(transactions)
    shops = true_breached_shops(transactions)
    rings = [g for g in groups if g.is_ring]
    matched, cards_found, shops_found = 0, 0, 0
    for ring_id, members in truth.items():
        best_group, best = None, 0.0
        for g in rings:
            overlap = len(members & set(g.cards)) / len(members | set(g.cards))
            if overlap > best:
                best_group, best = g, overlap
        matched += best >= 0.5
        cards_found += max((len(members & set(g.cards)) for g in rings), default=0)
        if best >= 0.5 and compromise is not None and ring_id in shops:
            candidates = compromise.get(best_group.group_id)
            if candidates is not None and len(candidates):
                shops_found += candidates.iloc[0]["merchant_id"] == shops[ring_id]
    ring_cards = set().union(*truth.values()) if truth else set()
    false_groups = sum(1 for g in rings if not set(g.cards) & ring_cards)
    result = {
        "true_rings": len(truth),
        "rings_found": int(matched),
        "victim_cards": len(ring_cards),
        "victim_cards_found": int(cards_found),
        "groups_flagged": len(rings),
        "false_ring_groups": int(false_groups),
        "linked_groups_checked": len(groups),
    }
    if compromise is not None:
        result["breached_shop_found"] = int(shops_found)
    return result
