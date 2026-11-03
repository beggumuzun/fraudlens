"""Shared, session-wide fixtures: one small dataset and one trained model for all tests."""

import pandas as pd
import pytest

from fraudlens.data import GeneratorConfig, generate_transactions
from fraudlens.features import build_features
from fraudlens.model import train


@pytest.fixture(scope="session")
def model_transactions() -> pd.DataFrame:
    config = GeneratorConfig(
        n_customers=400, n_days=30, seed=11, fraud_customer_share=0.1, travel_customer_share=0.05
    )
    return generate_transactions(config)


@pytest.fixture(scope="session")
def model_features(model_transactions) -> pd.DataFrame:
    return build_features(model_transactions)


@pytest.fixture(scope="session")
def model_bundle(model_features) -> dict:
    return train(model_features, max_fpr=0.01)
