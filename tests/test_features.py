import numpy as np
import pandas as pd
import pytest

from fraudlens.data import GeneratorConfig, generate_transactions, iter_transactions
from fraudlens.features import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    OnlineFeatureStore,
    OutOfOrderError,
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


def test_device_card_count_counts_other_cards_seen_before():
    df = pd.DataFrame(
        [
            make_txn(0, 0, card="A", channel="online", device="LAPTOP"),
            make_txn(1, 10, card="B", channel="online", device="LAPTOP"),
            make_txn(2, 20, card="A", channel="online", device="LAPTOP"),
            make_txn(3, 30, card="C", channel="online", device="LAPTOP"),
            make_txn(4, 40, card="D", channel="in_store"),
        ]
    )
    counts = build_features(df)["device_card_count"].tolist()
    # A first: nobody before. B: A before. A again: B before (A itself never counts).
    # C: A and B before. D paid in a shop: no device, so 0.
    assert counts == [0, 1, 1, 2, 0]


def test_ring_devices_light_up_the_network_feature(features):
    ring_rows = features[features["ring_id"].notna()]
    genuine = features[features["is_fraud"] == 0]
    assert ring_rows["device_card_count"].mean() > 10 * max(
        genuine["device_card_count"].mean(), 0.01
    )


# --- hand-made edge cases the generated data never produces ---


def assert_engines_agree(rows: list[dict]) -> pd.DataFrame:
    """Batch features of ``rows`` must equal the online engine fed in canonical order."""
    df = pd.DataFrame(rows)
    batch = build_features(df).set_index("transaction_id")
    store = OnlineFeatureStore()
    canonical = df.sort_values(["timestamp", "transaction_id"]).to_dict("records")
    online = pd.DataFrame(
        [store.process(txn) for txn in canonical],
        index=[txn["transaction_id"] for txn in canonical],
    ).loc[batch.index]
    for column in NUMERIC_FEATURES:
        np.testing.assert_allclose(
            online[column].to_numpy(), batch[column].to_numpy(), atol=1e-9, err_msg=column
        )
    return batch


def test_online_payments_without_a_device_agree():
    # Two online payments that sent no device ID: only the first is a "new device".
    batch = assert_engines_agree(
        [
            make_txn(0, 0, channel="online", device=None),
            make_txn(1, 10, channel="online", device=None),
            make_txn(2, 20, channel="online", device="PHONE"),
        ]
    )
    assert batch["is_new_device"].tolist() == [1, 0, 1]


def test_same_timestamp_ties_are_broken_by_transaction_id_in_both_engines():
    # T2 is listed first, but T1 comes first in the canonical (timestamp, id) order.
    batch = assert_engines_agree(
        [make_txn(5, 0, 2.0), make_txn(2, 30, 500.0), make_txn(1, 30, 1.0)]
    )
    assert batch.loc["T0000001", "n_prev_txns"] == 1
    assert batch.loc["T0000002", "n_prev_txns"] == 2
    # Same moment -> the earlier-id payment is NOT inside T2's velocity window [t-10m, t).
    assert batch.loc["T0000002", "txn_count_10m"] == 0


def test_online_engine_rejects_payments_that_arrive_out_of_order():
    store = OnlineFeatureStore()
    store.process(make_txn(2, 30))
    with pytest.raises(OutOfOrderError):
        store.process(make_txn(1, 10))  # older timestamp
    with pytest.raises(OutOfOrderError):
        store.process(make_txn(1, 30))  # same time, smaller id: batch would put it first
    with pytest.raises(OutOfOrderError):
        store.update(make_txn(1, 10))
    store.process(make_txn(3, 30))  # same time, larger id: fine
    store.process(make_txn(0, 40, card="OTHER"))  # another card has its own history
    assert store.compute(make_txn(9, 50))["n_prev_txns"] == 2


def test_zscore_is_not_fooled_by_rounding_noise():
    # Five identical £0.10 payments have no spread at all. Float cancellation used to
    # turn that into a tiny std and a z-score of +50 for a £0.11 payment.
    rows = [make_txn(i, i, 0.10) for i in range(5)] + [make_txn(5, 5, 0.11)]
    batch = assert_engines_agree(rows)
    assert batch["amount_zscore"].iloc[-1] == 0.0
    # A real spread still gives a real z-score.
    varied = [make_txn(i, i, a) for i, a in enumerate([10, 12, 11, 9, 10, 40])]
    assert assert_engines_agree(varied)["amount_zscore"].iloc[-1] > 10


def test_a_late_payment_on_another_card_is_rejected_too():
    # Shared devices link cards: B arriving late would change C's device count after the
    # fact, so the order check is global, not per card.
    store = OnlineFeatureStore()
    store.process(make_txn(0, 0, card="A", channel="online", device="LAPTOP"))
    store.process(make_txn(2, 20, card="C", channel="online", device="LAPTOP"))
    with pytest.raises(OutOfOrderError):
        store.process(make_txn(1, 10, card="B", channel="online", device="LAPTOP"))


def test_in_store_payments_never_make_a_missing_device_look_familiar():
    # A fraudster who strips the device ID is still on a "new device" the first time,
    # even if the card has been used in shops before.
    batch = assert_engines_agree(
        [
            make_txn(0, 0, channel="in_store"),
            make_txn(1, 10, channel="online", device=None),
            make_txn(2, 20, channel="online", device=None),
        ]
    )
    assert batch["is_new_device"].tolist() == [0, 1, 0]
    assert batch["new_device_24h"].tolist() == [0, 1, 1]


def test_numeric_transaction_ids_are_ordered_as_numbers():
    store = OnlineFeatureStore()
    rows = [make_txn(0, 0) | {"transaction_id": i} for i in (9, 10)]
    for txn in rows:
        store.process(txn)  # 9 then 10 at the same moment: fine (as strings "10" < "9")
    with pytest.raises(OutOfOrderError):
        store.process(make_txn(0, 0) | {"transaction_id": 8})
