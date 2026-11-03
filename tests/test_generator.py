import pandas as pd
import pytest

from fraudlens.data import COLUMNS, GeneratorConfig, generate_transactions, iter_transactions
from fraudlens.data.entities import CATEGORIES, FOREIGN_CITIES
from fraudlens.data.scenarios import FRAUD_SCENARIOS

# A small world with a high fraud share, so every scenario shows up and tests stay fast.
SMALL = GeneratorConfig(
    n_customers=300, n_days=20, seed=7, fraud_customer_share=0.1, travel_customer_share=0.05
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate_transactions(SMALL)


def test_schema(df):
    assert list(df.columns) == COLUMNS
    assert len(df) > 0


def test_same_seed_gives_identical_data(df):
    pd.testing.assert_frame_equal(df, generate_transactions(SMALL))


def test_different_seed_gives_different_data(df):
    other = generate_transactions(GeneratorConfig(**{**SMALL.__dict__, "seed": 8}))
    assert not df.equals(other)


def test_sorted_by_time_with_unique_ids(df):
    assert df["timestamp"].is_monotonic_increasing
    assert df["transaction_id"].is_unique


def test_values_are_valid(df):
    assert (df["amount"] >= 0.5).all()
    assert set(df["merchant_category"]) <= set(CATEGORIES)
    assert set(df["channel"]) == {"online", "in_store"}
    assert df["lat"].between(-90, 90).all() and df["lon"].between(-180, 180).all()
    assert set(df["is_fraud"].unique()) <= {0, 1}


def test_fraud_type_set_only_for_fraud(df):
    assert df.loc[df["is_fraud"] == 1, "fraud_type"].notna().all()
    assert df.loc[df["is_fraud"] == 0, "fraud_type"].isna().all()


def test_every_fraud_scenario_appears(df):
    assert set(df["fraud_type"].dropna()) == set(FRAUD_SCENARIOS)


def test_no_fraud_during_warmup(df):
    warmup_end = pd.Timestamp(SMALL.start_date) + pd.Timedelta(days=SMALL.warmup_days)
    assert (df.loc[df["is_fraud"] == 1, "timestamp"] >= warmup_end).all()


def test_card_testing_is_a_fast_burst_of_tiny_payments(df):
    attacks = df[df["fraud_type"] == "card_testing"]
    for _, attack in attacks.groupby("customer_id"):
        assert (attack["amount"] <= 3.0).sum() >= 5
        assert attack["timestamp"].max() - attack["timestamp"].min() < pd.Timedelta(hours=2)


def test_genuine_foreign_transactions_exist(df):
    """Hard negatives: not every foreign transaction is fraud."""
    abroad = df[df["merchant_city"].isin(FOREIGN_CITIES)]
    assert (abroad["is_fraud"] == 0).any()


def test_default_fraud_rate_is_realistic():
    data = generate_transactions(GeneratorConfig(n_customers=1_000))
    rate = data["is_fraud"].mean()
    assert 0.0005 < rate < 0.01  # real card fraud is well under 1% of transactions


def test_stream_hides_labels_by_default(df):
    first = next(iter_transactions(df))
    assert "is_fraud" not in first and "fraud_type" not in first
    assert "is_fraud" in next(iter_transactions(df, include_labels=True))


@pytest.mark.parametrize(
    "kwargs",
    [{"n_customers": 0}, {"n_days": 5, "warmup_days": 7}, {"fraud_customer_share": 1.5}],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GeneratorConfig(**kwargs)


# --- fraud rings and households ---


def test_ring_labels_only_on_ring_fraud(df):
    assert df.loc[df["ring_id"].notna(), "is_fraud"].eq(1).all()
    online = df["fraud_type"].isin(["card_testing", "account_takeover"])
    assert (
        df.loc[df["ring_id"].notna(), "fraud_type"].isin(["card_testing", "account_takeover"]).all()
    )
    assert df.loc[online, "ring_id"].notna().any()  # some online attacks are ring work
    assert df.loc[df["fraud_type"] == "stolen_card_spree", "ring_id"].isna().all()


def test_a_ring_reuses_its_devices_across_victims(df):
    rings = df[df["ring_id"].notna()].groupby("ring_id")
    for _, ring in rings:
        assert ring["card_id"].nunique() >= 2
        assert ring["device_id"].nunique() <= 3  # a small device pool shared by all victims


def test_some_genuine_devices_are_shared_by_families(df):
    genuine = df[(df["is_fraud"] == 0) & df["device_id"].notna()]
    cards_per_device = genuine.groupby("device_id")["card_id"].nunique()
    assert (cards_per_device >= 2).any()
