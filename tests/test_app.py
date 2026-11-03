"""The app: the demo world's maths, and every screen rendering and reacting without errors.

Screens are driven headlessly with Streamlit's AppTest, on a small demo world.
"""

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from fraudlens.app.state import SMALL_CONFIG
from fraudlens.app.world import build_demo_world, features_frame

TIMEOUT = 300


@pytest.fixture(scope="module")
def world():
    return build_demo_world(SMALL_CONFIG)


@pytest.fixture(autouse=True)
def small_world(monkeypatch):
    monkeypatch.setenv("FRAUDLENS_DEMO_SIZE", "small")


# --- the demo world ---


def test_risk_is_a_percentile_and_alerts_start_near_99_9(world):
    risk = world.risk([0.0, world.threshold, 1.0])
    assert risk[0] == 0 and risk[2] == 100
    assert 99.0 <= risk[1] <= 100  # the threshold allows ~1 false alarm per 1,000


def test_score_flags_exactly_the_payments_above_threshold(world):
    sample = world.features.sample(300, random_state=1)
    scored = world.score(sample)
    assert (scored["alert"] == (scored["score"] >= world.threshold)).all()
    assert all(r for r in scored.loc[scored["alert"], "reasons"]) or not scored["alert"].any()


def test_live_store_starts_exactly_at_the_test_period(world):
    first = world.test_transactions.iloc[0].to_dict()
    online = features_frame([world.store_at_test_start.compute(first)])
    batch = world.features.set_index("transaction_id").loc[[first["transaction_id"]]]
    np.testing.assert_allclose(
        online.drop(columns="merchant_category").to_numpy(dtype=float),
        batch[online.columns.drop("merchant_category")].to_numpy(dtype=float),
        atol=1e-6,
    )


def test_snapshot_leaves_the_shared_history_untouched(world):
    """A what-if payment on a snapshot must not leak into the shared store, including the
    device links between cards (the part the old test never exercised)."""
    card, other = world.cards.index[0], world.cards.index[1]
    when = world.transactions["timestamp"].max() + pd.Timedelta(hours=1)
    payment = world.test_transactions.iloc[0].to_dict() | {
        "card_id": card,
        "channel": "online",
        "device_id": "WHAT_IF_DEVICE",
        "timestamp": when,
        "transaction_id": "SIM001",
    }
    probe = payment | {"card_id": other, "timestamp": when + pd.Timedelta(minutes=5)}
    before_card = world.store_full.compute(payment)
    before_other = world.store_full.compute(probe)

    copy_ = world.store_full.snapshot([card])
    copy_.process(payment)
    assert copy_.compute(probe)["device_card_count"] == 1  # the copy saw the link...

    assert world.store_full.compute(payment) == before_card  # ...the shared store did not
    assert world.store_full.compute(probe) == before_other
    assert before_other["device_card_count"] == 0


# --- the screens ---


def run_page(name: str) -> AppTest:
    source = f"import importlib\nimportlib.import_module('fraudlens.app.views.{name}').render()\n"
    at = AppTest.from_string(source, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


def test_live_monitor_scores_payments_when_running():
    at = run_page("live")
    at.toggle(key="live_running").set_value(True).run()
    assert not at.exception, at.exception
    panel = " ".join(m.value for m in at.markdown)
    assert "Payments scored" in panel
    assert '<div class="fl-kpi-value">0</div><div class="fl-kpi-label">Payments scored' not in panel


def test_test_a_payment_alerts_on_night_shopping_abroad():
    at = run_page("test_payment")
    next(b for b in at.button if b.label == "Night shopping abroad").click().run()
    assert not at.exception, at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Held for review" in page, "expected an alert for night jewellery shopping abroad"


def verdict(at: AppTest) -> str:
    page = " ".join(m.value for m in at.markdown)
    if "Caught on the very first payment" in page:
        return "first"
    if "Caught on payment" in page:
        return "later"
    assert "Missed" in page, "a verdict is always shown"
    return "missed"


def expected_verdict(result: pd.DataFrame) -> str:
    alerts = result.index[result["alert"]]
    if len(alerts) == 0:
        return "missed"
    return "first" if alerts[0] == 0 else "later"


def launch(attack: str | None = None) -> AppTest:
    at = run_page("simulator")
    at.toggle[0].set_value(False).run()  # no step-by-step animation in tests
    if attack is not None:
        at.radio(key="sim_attack").set_value(attack).run()
    next(b for b in at.button if b.label == "Launch attack").click().run()
    assert not at.exception, at.exception
    return at


def test_attack_simulator_shows_the_verdict_the_model_gives(world):
    """The page's verdict must be exactly what the model did with that attack."""
    from fraudlens.app.views.simulator import _run_attack

    at = launch()
    card, attack, seed = at.session_state["sim_key"]
    assert verdict(at) == expected_verdict(_run_attack(world, card, attack, seed))


def test_simulator_can_replay_a_known_ring(world):
    from fraudlens.app.views.simulator import RING_ATTACK, _run_attack

    at = launch(RING_ATTACK)
    card, attack, seed = at.session_state["sim_key"]
    assert attack == RING_ATTACK
    assert verdict(at) == expected_verdict(_run_attack(world, card, attack, seed))


def test_ring_attacks_are_caught(world):
    """Fresh ring attacks on 15 clean cards of the small test world: all are caught.

    (This world's model learns from far fewer attacks than the full demo world's, so
    some are caught on their second or third payment rather than the first.)
    """
    from fraudlens.app.views.simulator import RING_ATTACK, _run_attack

    clean = world.cards[world.cards["frauds"] == 0].index[:15]
    results = [_run_attack(world, card, RING_ATTACK, seed=i) for i, card in enumerate(clean)]
    assert all(r["alert"].any() for r in results)


def test_model_performance_renders_the_trade_off():
    at = run_page("performance")
    page = " ".join(m.value for m in at.markdown)
    assert "Attacks caught" in page and "Alerts per day" in page


def test_fraud_rings_page_finds_rings_and_a_leak():
    at = run_page("rings")
    page = " ".join(m.value for m in at.markdown)
    assert "Rings found" in page
    assert "likely point of compromise" in page


def test_a_known_ring_device_gives_the_network_feature_its_value(world):
    """The ring's device was seen on other cards, so the network feature is already non-zero
    on the attack's very first payment. (Whether it changes the SCORE is a separate
    question, answered by the ablation in the README: on this data, barely.)"""
    from fraudlens.app.world import features_frame, strip_labels
    from fraudlens.data.scenarios import card_testing

    card = world.cards[world.cards["frauds"] == 0].index[0]
    actor = next(iter(world.ring_actors.values()))
    records = card_testing(
        world.customer(card),
        world.config.n_days,
        world.directory,
        np.random.default_rng(0),
        actor=actor,
    )
    records.sort(key=lambda r: r["offset_s"])
    store = world.store_full.snapshot([card])
    start = pd.Timestamp(world.config.start_date)
    first = records[0]
    txn = strip_labels({k: v for k, v in first.items() if k != "offset_s"})
    txn["timestamp"] = start + pd.Timedelta(seconds=first["offset_s"])
    features = features_frame([store.process(txn)])
    assert features["device_card_count"].iloc[0] >= 2
