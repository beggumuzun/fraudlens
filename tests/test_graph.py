import random
from collections import defaultdict

import pandas as pd
import pytest

from fraudlens.data import GeneratorConfig, generate_transactions
from fraudlens.graph import (
    UnionFind,
    find_rings,
    linked_groups,
    point_of_compromise,
    score_against_truth,
)

# --- union-find ---


def test_union_find_basics():
    uf = UnionFind(["a", "b", "c", "d"])
    assert not uf.connected("a", "b")
    uf.union("a", "b")
    uf.union("c", "d")
    assert uf.connected("a", "b") and uf.connected("c", "d")
    assert not uf.connected("a", "c")
    uf.union("b", "d")
    assert uf.connected("a", "c")
    assert sorted(len(g) for g in uf.groups()) == [4]


def test_union_find_adds_unknown_items_on_the_fly():
    uf = UnionFind()
    uf.union("x", "y")
    assert "x" in uf and "y" in uf and len(uf) == 2


def _components_by_bfs(n, edges):
    graph = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    seen, components = set(), []
    for start in range(n):
        if start in seen:
            continue
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(graph[node] - component)
        seen |= component
        components.append(frozenset(component))
    return set(components)


@pytest.mark.parametrize("seed", range(5))
def test_union_find_matches_breadth_first_search_on_random_graphs(seed):
    rng = random.Random(seed)
    n = 300
    edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(220)]
    uf = UnionFind(range(n))
    for a, b in edges:
        uf.union(a, b)
    assert {frozenset(g) for g in uf.groups()} == _components_by_bfs(n, edges)


# --- rings on a hand-made example ---


def _pay(card, device, day, merchant="M1", category="clothing"):
    return {
        "card_id": card,
        "device_id": device,
        "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
        "merchant_id": merchant,
        "merchant_category": category,
        "merchant_city": "ONLINE",
    }


def test_rings_are_separated_from_families_by_model_alerts():
    tx = pd.DataFrame(
        [
            # a family of three sharing one tablet
            _pay("F1", "TABLET", 1), _pay("F2", "TABLET", 2), _pay("F3", "TABLET", 3),
            # a ring rotating two devices across three victims (each victim also has a phone)
            _pay("V1", "PHONE1", 1), _pay("V1", "RING_A", 9), _pay("V1", "RING_B", 9),
            _pay("V2", "PHONE2", 1), _pay("V2", "RING_B", 10),
            _pay("V3", "PHONE3", 1), _pay("V3", "RING_A", 11),
        ]
    )  # fmt: skip
    groups = {tuple(g.cards): g for g in find_rings(tx, flagged_cards={"V1", "V2", "V3"})}
    assert set(groups) == {("F1", "F2", "F3"), ("V1", "V2", "V3")}
    assert groups[("V1", "V2", "V3")].is_ring
    assert not groups[("F1", "F2", "F3")].is_ring  # same structure, but nobody was flagged


def test_single_card_devices_do_not_form_groups():
    tx = pd.DataFrame([_pay("A", "P1", 1), _pay("B", "P2", 1)])
    assert linked_groups(tx) == []


def test_point_of_compromise_finds_the_shop_all_victims_used():
    rows = []
    for i in range(20):  # 20 ordinary customers, a few of whom use the breached shop too
        rows.append(_pay(f"C{i}", None, 1, merchant="BIG_SHOP"))
        if i < 3:
            rows.append(_pay(f"C{i}", None, 2, merchant="BREACHED"))
    for v in ("V1", "V2", "V3", "V4"):
        rows.append(_pay(v, None, 1, merchant="BIG_SHOP"))
        rows.append(_pay(v, None, 3, merchant="BREACHED"))
    tx = pd.DataFrame(rows)
    first_alert = {v: pd.Timestamp("2026-01-10") for v in ("V1", "V2", "V3", "V4")}
    table = point_of_compromise(tx, first_alert)
    top = table.iloc[0]
    assert top["merchant_id"] == "BREACHED"
    assert top["victims"] == 4 and top["lift"] > 3  # BIG_SHOP: all victims too, but lift ~1


def test_payments_after_the_first_alert_are_ignored():
    tx = pd.DataFrame(
        [_pay("V1", None, 5, merchant="LATER"), _pay("V1", None, 1, merchant="EARLIER")]
    )
    table = point_of_compromise(tx, {"V1": pd.Timestamp("2026-01-04")})
    assert table["merchant_id"].tolist() == ["EARLIER"]


# --- rings in generated data (graph logic checked with perfect alerts) ---


@pytest.fixture(scope="module")
def ring_world():
    config = GeneratorConfig(
        n_customers=600, n_days=30, seed=5, fraud_customer_share=0.08, victims_per_ring=6
    )
    return generate_transactions(config)


def test_every_generated_ring_is_recovered(ring_world):
    fraud_cards = set(ring_world.loc[ring_world["is_fraud"] == 1, "card_id"])
    groups = find_rings(ring_world, flagged_cards=fraud_cards)
    score = score_against_truth(groups, ring_world)
    assert score["true_rings"] >= 2
    assert score["rings_found"] == score["true_rings"]
    assert score["victim_cards_found"] == score["victim_cards"]
    assert score["false_ring_groups"] == 0


def test_breached_shop_is_found_for_every_ring(ring_world):
    fraud = ring_world[ring_world["ring_id"].notna()]
    for _, victims in fraud.groupby("ring_id"):
        first_alert = victims.groupby("card_id")["timestamp"].min().to_dict()
        top = point_of_compromise(ring_world, first_alert).iloc[0]
        assert top["victims"] == len(first_alert)  # every victim paid there beforehand
        assert top["lift"] > 2
