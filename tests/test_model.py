import numpy as np
import pandas as pd
import pytest

from fraudlens.model import (
    evaluate,
    load_bundle,
    predict_scores,
    save_bundle,
    threshold_at_fpr,
    time_split,
)


@pytest.fixture(scope="module")
def features(model_features) -> pd.DataFrame:
    return model_features


@pytest.fixture(scope="module")
def bundle(model_bundle) -> dict:
    return model_bundle


# --- threshold and metrics, checked by hand ---


def test_threshold_respects_false_alarm_budget():
    y = np.array([0] * 10 + [1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.95])
    # 10% of 10 genuine payments = 1 false alarm allowed -> only the 1.0 genuine is flagged
    threshold = threshold_at_fpr(y, scores, max_fpr=0.1)
    assert 0.9 < threshold <= 0.95
    assert ((scores >= threshold) & (y == 0)).sum() == 1
    # No false alarms allowed -> threshold sits above every genuine score
    assert threshold_at_fpr(y, scores, max_fpr=0.0) > 1.0


def test_evaluate_counts_money_and_attacks():
    scored = pd.DataFrame(
        {
            "card_id": ["A", "A", "B", "C", "C"],
            "fraud_type": ["card_testing", "card_testing", "account_takeover", None, None],
            "is_fraud": [1, 1, 1, 0, 0],
            "amount": [1.0, 99.0, 100.0, 10.0, 20.0],
            "score": [0.2, 0.9, 0.1, 0.95, 0.05],
        }
    )
    m = evaluate(scored, threshold=0.5)
    assert m["recall"] == pytest.approx(1 / 3)  # 1 of 3 fraud payments flagged
    assert m["value_recall"] == pytest.approx(99 / 200)  # £99 of £200 stopped
    assert m["attack_recall"] == pytest.approx(0.5)  # card A caught, card B missed
    assert m["precision"] == pytest.approx(0.5)  # 2 alerts, 1 genuine
    assert m["false_positive_rate"] == pytest.approx(0.5)


# --- split ---


def test_time_split_is_ordered_and_complete(features):
    split = time_split(features)
    assert split.train["timestamp"].max() < split.valid["timestamp"].min()
    assert split.valid["timestamp"].max() < split.test["timestamp"].min()
    assert len(split.train) + len(split.valid) + len(split.test) == len(features)


def test_time_split_rejects_bad_fractions(features):
    with pytest.raises(ValueError):
        time_split(features, valid_frac=0.6, test_frac=0.5)


# --- training end to end ---


def test_models_learn_something_useful(bundle):
    base_rate = bundle["split"]["test"]["frauds"] / bundle["split"]["test"]["rows"]
    for name, candidate in bundle["candidates"].items():
        # Random guessing would score about the base rate.
        assert candidate["test"]["average_precision"] > 20 * base_rate, name


def test_best_model_is_chosen_on_validation_not_test(bundle):
    valid_ap = {n: c["valid"]["average_precision"] for n, c in bundle["candidates"].items()}
    assert bundle["model_name"] == max(valid_ap, key=valid_ap.get)


def test_threshold_keeps_validation_false_alarms_in_budget(bundle):
    best = bundle["candidates"][bundle["model_name"]]
    assert best["valid"]["false_positive_rate"] <= bundle["max_fpr"]


def test_saved_model_gives_identical_scores(bundle, features, tmp_path):
    save_bundle(bundle, tmp_path / "model.joblib", tmp_path / "metrics.json")
    loaded = load_bundle(tmp_path / "model.joblib")
    sample = features.head(500)
    np.testing.assert_array_equal(
        predict_scores(bundle["model"], sample), predict_scores(loaded["model"], sample)
    )
    assert loaded["threshold"] == bundle["threshold"]
    assert (tmp_path / "metrics.json").exists()
