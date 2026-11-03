"""The app: the demo world's maths, and every screen rendering and reacting without errors.

Screens are driven headlessly with Streamlit's AppTest, on a small demo world.
"""

import numpy as np
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
    card = world.cards.index[0]
    before = world.store_full.compute(world.test_transactions.iloc[0].to_dict() | {"card_id": card})
    copy_ = world.store_full.snapshot([card])
    copy_.update(world.test_transactions.iloc[0].to_dict() | {"card_id": card})
    after = world.store_full.compute(world.test_transactions.iloc[0].to_dict() | {"card_id": card})
    assert before == after


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


def test_attack_simulator_reports_an_outcome():
    at = run_page("simulator")
    at.toggle[0].set_value(False).run()  # no step-by-step animation in tests
    next(b for b in at.button if b.label == "Launch attack").click().run()
    assert not at.exception, at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Caught" in page or "Missed" in page  # a verdict is always shown


def test_model_performance_renders_the_trade_off():
    at = run_page("performance")
    page = " ".join(m.value for m in at.markdown)
    assert "Attacks caught" in page and "Alerts per day" in page
