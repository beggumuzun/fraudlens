import numpy as np
import pandas as pd
import pytest

from fraudlens.data import GeneratorConfig, generate_transactions, iter_transactions
from fraudlens.features import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    OnlineFeatureStore,
    build_features,
)
from fraudlens.features.definitions import LABEL_COLUMNS, NO_HISTORY_GAP_S

BRISTOL = (51.4545, -2.5879)
LAGOS = (6.5244, 3.3792)


def make_txn(
    i,
    minutes,
    amount=10.0,
    *,
    card="CARD1",
    channel="in_store",
    device=None,
    location=BRISTOL,
    category="grocery",
):
    return {
        "transaction_id": f"T{i:07d}",
        "timestamp": pd.Timestamp("2026-01-10 12:00") + pd.Timedelta(minutes=minutes),
        "card_id": card,
        "merchant_category": category,
        "channel": channel,
        "amount": amount,
        "lat": location[0],
        "lon": location[1],
        "device_id": device,
    }


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    config = GeneratorConfig(
        n_customers=200, n_days=20, seed=3, fraud_customer_share=0.1, travel_customer_share=0.05
    )
    return generate_transactions(config)


@pytest.fixture(scope="module")
def features(transactions) -> pd.DataFrame:
    return build_features(transactions)


def test_labels_are_never_features():
    assert not set(LABEL_COLUMNS) & set(FEATURE_COLUMNS)


def test_one_row_per_transaction_in_original_order(transactions, features):
    assert features["transaction_id"].tolist() == transactions["transaction_id"].tolist()
    assert features[NUMERIC_FEATURES].notna().all().all()


def test_hand_computed_example():
    """Three payments of £10, £20, £30 at 12:00, 12:05 and 12:15, checked by hand."""
    df = pd.DataFrame([make_txn(0, 0, 10), make_txn(1, 5, 20), make_txn(2, 15, 30)])
    first, _, third = build_features(df).to_dict("records")

    assert first["n_prev_txns"] == 0
    assert first["seconds_since_last"] == NO_HISTORY_GAP_S
    assert first["txn_count_24h"] == 0 and first["amount_to_avg_ratio"] == 1.0

    assert third["n_prev_txns"] == 2
    assert third["seconds_since_last"] == 600
    assert third["txn_count_10m"] == 1  # window [12:05, 12:15) contains only the 12:05 one
    assert third["txn_count_1h"] == 2
    assert third["amount_sum_1h"] == 30
    assert third["amount_to_avg_ratio"] == pytest.approx(2.0)  # 30 / mean(10, 20)
    assert third["amount_zscore"] == pytest.approx(15 / np.std([10, 20], ddof=1))
    assert third["dist_from_last_km"] == pytest.approx(0.0)


def test_impossible_travel_and_new_device():
    """Paying in Bristol, then online from Lagos an hour later on an unknown device."""
    df = pd.DataFrame(
        [
            make_txn(0, 0, 20, channel="online", device="PHONE"),
            make_txn(
                1,
                60,
                900,
                channel="online",
                device="STRANGER",
                location=LAGOS,
                category="electronics",
            ),
        ]
    )
    home, attack = build_features(df).to_dict("records")
    assert home["is_new_device"] == 1  # first ever use of the customer's own phone
    assert attack["is_new_device"] == 1
    assert attack["is_outside_uk"] == 1
    assert attack["is_new_category"] == 1
    assert attack["dist_from_last_km"] > 4_500
    assert attack["speed_from_last_kmh"] > 4_500  # no plane flies that fast


def test_no_leakage_from_the_future(transactions, features):
    """Deleting all future payments must not change a single past feature value."""
    cutoff = transactions["timestamp"].quantile(0.6)
    past_only = build_features(transactions[transactions["timestamp"] < cutoff])
    same_rows = features[features["timestamp"] < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(past_only, same_rows)


def test_fraud_signals_are_visible(features):
    card_testing = features[features["fraud_type"] == "card_testing"]
    genuine = features[features["is_fraud"] == 0]
    assert card_testing["small_txn_count_10m"].mean() > 10 * genuine["small_txn_count_10m"].mean()

    takeover = features[features["fraud_type"] == "account_takeover"]
    # is_new_device only fires on the attacker's FIRST payment; new_device_24h covers them all
    assert takeover["new_device_24h"].mean() > 0.9
    assert takeover["amount_to_avg_ratio"].median() > 3


def test_online_engine_matches_batch_engine(transactions, features):
    """Replaying the stream one payment at a time must give the same numbers as batch."""
    store = OnlineFeatureStore()
    online = pd.DataFrame([store.process(txn) for txn in iter_transactions(transactions)])

    assert (online["merchant_category"] == features["merchant_category"]).all()
    for column in NUMERIC_FEATURES:
        np.testing.assert_allclose(
            online[column].to_numpy(),
            features[column].to_numpy(),
            rtol=1e-9,
            atol=1e-6,
            err_msg=column,
        )


def test_online_compute_does_not_change_state():
    store = OnlineFeatureStore()
    txn = make_txn(0, 0, 10)
    assert store.compute(txn) == store.compute(txn)
    assert len(store) == 0
    store.process(txn)
    assert store.compute(make_txn(1, 1, 10))["n_prev_txns"] == 1
