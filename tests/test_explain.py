import numpy as np
import pandas as pd
import pytest

from fraudlens.explain import FEATURE_GROUPS, ReasonExplainer, describe, reference_profile
from fraudlens.explain.cli import explain_test_alerts
from fraudlens.features.definitions import FEATURE_COLUMNS
from fraudlens.model import predict_scores


class StubModel:
    """A fake model whose log-odds we know exactly, to check the Shapley maths by hand."""

    def __init__(self, formula):
        self.formula = formula

    def decision_function(self, x: pd.DataFrame) -> np.ndarray:
        return self.formula(x).astype(float).to_numpy()


def stub_explainer(formula) -> ReasonExplainer:
    reference = {col: 0.0 for col in FEATURE_COLUMNS}
    reference["merchant_category"] = "grocery"
    return ReasonExplainer(StubModel(formula), reference, threshold=0.5)


def suspicious_payment() -> pd.DataFrame:
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row.update(merchant_category="grocery", is_new_device=1.0, is_outside_uk=1.0)
    return pd.DataFrame([row])


def phi_by_group(explainer, rows) -> dict:
    _, phi, _ = explainer.shapley(rows)
    return dict(zip(FEATURE_GROUPS, phi[:, 0], strict=True))


# --- the maths ---


def test_every_feature_is_in_exactly_one_group():
    grouped = [col for cols in FEATURE_GROUPS.values() for col in cols]
    assert sorted(grouped) == sorted(FEATURE_COLUMNS)


def test_additive_model_gets_its_own_weights_back():
    """log-odds = 2 * new_device + 1 * outside_uk  ->  device 2, location 1, others 0."""
    explainer = stub_explainer(lambda x: 2 * x["is_new_device"] + x["is_outside_uk"])
    phi = phi_by_group(explainer, suspicious_payment())
    assert phi["device"] == pytest.approx(2.0)
    assert phi["location"] == pytest.approx(1.0)
    assert sum(abs(v) for g, v in phi.items() if g not in ("device", "location")) < 1e-12


def test_interaction_credit_is_shared_fairly():
    """log-odds = 3 only if new device AND abroad. Each signal earns half: 1.5 each.

    Switching off one group at a time would wrongly give BOTH groups the full 3.
    """
    explainer = stub_explainer(lambda x: 3 * (x["is_new_device"] * x["is_outside_uk"]))
    phi = phi_by_group(explainer, suspicious_payment())
    assert phi["device"] == pytest.approx(1.5)
    assert phi["location"] == pytest.approx(1.5)


def test_reasons_are_ranked_shares():
    explainer = stub_explainer(lambda x: 2 * x["is_new_device"] + x["is_outside_uk"])
    reasons = explainer.explain(suspicious_payment())[0]["reasons"]
    assert [r["group"] for r in reasons] == ["device", "location"]
    assert [r["share"] for r in reasons] == pytest.approx([2 / 3, 1 / 3], abs=1e-3)


# --- the real model ---


def test_shapley_values_add_up_exactly(model_features, model_bundle):
    explainer = ReasonExplainer.from_bundle(model_bundle)
    sample = model_features[model_features["is_fraud"] == 1].head(20)
    log_odds, phi, reference = explainer.shapley(sample)
    np.testing.assert_allclose(log_odds, reference + phi.sum(axis=0), atol=1e-9)


def test_explained_scores_match_model_scores(model_features, model_bundle):
    explainer = ReasonExplainer.from_bundle(model_bundle)
    sample = model_features.sample(50, random_state=0)
    scores = [e["score"] for e in explainer.explain(sample)]
    np.testing.assert_allclose(scores, predict_scores(model_bundle["model"], sample), atol=1e-9)


def test_typical_payment_has_no_reasons(model_bundle):
    explainer = ReasonExplainer.from_bundle(model_bundle)
    typical = pd.DataFrame([model_bundle["reference"]])
    assert explainer.explain(typical)[0]["reasons"] == []


def test_attacks_are_explained_by_the_right_signals(model_features, model_bundle):
    table, examples = explain_test_alerts(model_features, model_bundle)
    assert examples, "expected some alerts in the test period"
    card_testing = table.loc["card_testing"]
    assert (
        card_testing.get("device", 0) + card_testing.get("velocity", 0)
    ) / card_testing.sum() >= 0.7


def test_old_bundle_without_reference_needs_features(model_bundle, model_features):
    old = {k: v for k, v in model_bundle.items() if k != "reference"}
    with pytest.raises(ValueError):
        ReasonExplainer.from_bundle(old)
    explainer = ReasonExplainer.from_bundle(old, model_features)
    assert explainer.reference == reference_profile(model_features)


# --- the sentences ---


def test_sentences_use_real_values():
    values = {col: 0.0 for col in FEATURE_COLUMNS}
    values.update(
        {
            "speed_from_last_kmh": 2_400.0,
            "dist_from_last_km": 4_800.0,
            "small_txn_count_10m": 6.0,
            "txn_count_1h": 1.0,
            "amount": 0.75,
            "amount_to_avg_ratio": 0.03,
            "merchant_category": "digital_goods",
            "is_new_category": 1.0,
        }
    )
    row = pd.Series(values, dtype=object)
    assert "Impossible travel" in describe("location", row)
    assert "4,800 km" in describe("location", row)
    assert describe("velocity", row).startswith("6 tiny payments")
    assert "Unusually small payment of £0.75" in describe("amount", row)
    assert describe("merchant", row) == "First ever digital goods purchase on this card"

    row["small_txn_count_10m"] = 0.0
    assert describe("velocity", row).startswith("1 payment and")  # singular, not "1 payments"


def test_shared_device_is_only_blamed_when_it_moves_the_score():
    """A ring payment is BOTH from a new device and from a device seen on 12 other cards.

    If the model only reacts to the new device, the explanation must not claim that the
    shared device gave the payment away (the bug this group split fixed).
    """
    row = suspicious_payment().assign(device_card_count=12.0, is_online=1.0)

    new_device_only = stub_explainer(lambda x: 4 * x["is_new_device"])
    groups = [r["group"] for r in new_device_only.explain(row)[0]["reasons"]]
    assert groups == ["device"]

    network_only = stub_explainer(lambda x: 0.5 * x["device_card_count"])
    reasons = network_only.explain(row)[0]["reasons"]
    assert [r["group"] for r in reasons] == ["network"]
    assert reasons[0]["text"] == "This device has already been used with 12 other cards"
