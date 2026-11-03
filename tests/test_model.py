import numpy as np
import pandas as pd
import pytest

from fraudlens.bench import run as run_benchmark
from fraudlens.model import (
    evaluate,
    load_bundle,
    predict_scores,
    save_bundle,
    threshold_at_fpr,
    time_split,
)
from fraudlens.model.ablation import ring_attack_recall
from fraudlens.model.ablation import run as run_ablation


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


def test_baseline_log_transforms_only_non_negative_skewed_features(features):
    """log1p is only safe on values >= 0: every skewed feature must stay non-negative."""
    from fraudlens.model.train import SKEWED_FEATURES

    assert (features[SKEWED_FEATURES] >= 0).all().all()


def test_baseline_can_be_saved_and_reloaded(features, tmp_path):
    import joblib

    from fraudlens.model.train import make_logistic_regression

    model = make_logistic_regression(seed=0).fit(features, features["is_fraud"])
    joblib.dump(model, tmp_path / "baseline.joblib")
    reloaded = joblib.load(tmp_path / "baseline.joblib")
    sample = features.head(50)
    np.testing.assert_allclose(predict_scores(model, sample), predict_scores(reloaded, sample))


def test_a_model_can_be_served_by_an_earlier_decision(features):
    from fraudlens.model import train

    bundle = train(features, max_fpr=0.01, importance_repeats=1, serve="logistic_regression")
    assert bundle["model_name"] == "logistic_regression"
    assert bundle["selected_by"] == "fixed"
    with pytest.raises(ValueError):
        train(features, importance_repeats=1, serve="random_forest")


# --- ablation and benchmark tooling ---


def test_ring_attack_recall_by_hand():
    scored = pd.DataFrame(
        {
            "card_id": ["A", "A", "B", "B", "C"],
            "timestamp": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 10:05"] * 2 + ["2026-01-02 10:00"]
            ),
            "ring_id": ["R1", "R1", "R1", "R1", None],
            "score": [0.9, 0.2, 0.1, 0.8, 0.99],
        }
    )
    # A: first payment caught. B: caught only on its second. C is not a ring attack.
    assert ring_attack_recall(scored, 0.5) == {
        "ring_attacks": 2,
        "caught": 1.0,
        "caught_on_first_payment": 0.5,
    }


def test_ablation_trains_both_variants(features):
    result = run_ablation(features, max_fpr=0.01)
    assert result["feature"] == "device_card_count"
    for variant in ("with_feature", "without_feature"):
        assert 0 <= result[variant]["average_precision"] <= 1
    with pytest.raises(ValueError):
        run_ablation(features, feature="not_a_feature")


def test_benchmark_times_every_live_step(model_transactions, features, bundle):
    result = run_benchmark(model_transactions, features, bundle, n_payments=200, n_explain=5)
    for step in ("features", "score", "explain"):
        assert result[step]["runs"] > 0
        assert 0 < result[step]["p50_ms"] <= result[step]["p99_ms"]
