"""The demo world behind the app: data, trained model, explainer and live feature engines.

Built once when the app starts (about half a minute) and shared by every visitor, so the
app runs anywhere, including a fresh cloud server, with no data or model files needed.

Timeline of the world:

    |------ train ------|-- validation --|------------- test -------------|
                                         ^                                ^
                              store_at_test_start                    store_full

* The live monitor replays the test period (payments the model never saw) through a
  copy of ``store_at_test_start``, exactly as a production system would receive them.
* "Test a payment" and the attack simulator add new payments after the end of the data,
  on top of ``store_full``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudlens.data.entities import Customer, MerchantDirectory
from fraudlens.data.generator import (
    LABEL_COLUMNS,
    GeneratorConfig,
    build_world,
    generate_transactions,
    iter_transactions,
)
from fraudlens.data.scenarios import Actor
from fraudlens.explain.reasons import ReasonExplainer
from fraudlens.features.batch import build_features
from fraudlens.features.definitions import FEATURE_COLUMNS
from fraudlens.features.online import OnlineFeatureStore
from fraudlens.graph import LinkedGroup, ring_report
from fraudlens.model.split import TimeSplit, time_split
from fraudlens.model.train import predict_scores, train

APP_CONFIG = GeneratorConfig(n_customers=2_000, n_days=45, seed=7)
SERVED_MODEL = "gradient_boosting"  # winner of the full evaluation in the README


@dataclass
class DemoWorld:
    config: GeneratorConfig
    customers: list[Customer]
    directory: MerchantDirectory
    transactions: pd.DataFrame
    features: pd.DataFrame
    split: TimeSplit
    bundle: dict
    explainer: ReasonExplainer
    genuine_valid_scores: np.ndarray  # sorted, used to turn scores into risk percentiles
    valid_scored: pd.DataFrame
    test_scored: pd.DataFrame
    test_transactions: pd.DataFrame
    store_at_test_start: OnlineFeatureStore
    store_full: OnlineFeatureStore
    cards: pd.DataFrame  # one row per card: profile used by the app pages
    alerts: pd.DataFrame  # every payment the model flags, across the whole history
    groups: list[LinkedGroup]  # cards linked through shared devices
    compromise: dict[str, pd.DataFrame]  # ring group id -> likely breached shops
    ring_actors: dict[str, Actor]  # group id -> the ring's devices, for new simulated attacks

    @property
    def threshold(self) -> float:
        return self.bundle["threshold"]

    def risk(self, scores) -> np.ndarray:
        """Risk 0-100: the share of genuine payments that look LESS suspicious than this one.

        Raw model scores are tiny probabilities (fraud is rare), which are hard to read.
        "Riskier than 99.95% of genuine payments" is not.
        """
        ranks = np.searchsorted(self.genuine_valid_scores, np.asarray(scores), side="left")
        return 100 * ranks / len(self.genuine_valid_scores)

    def customer(self, card_id: str) -> Customer:
        return self.customers[int(card_id.removeprefix("CARD"))]

    def score(self, features: pd.DataFrame, explain: bool = True) -> pd.DataFrame:
        """Score feature rows; add risk, alert flag and (optionally) reasons."""
        scores = predict_scores(self.bundle["model"], features)
        out = pd.DataFrame(
            {
                "score": scores,
                "risk": self.risk(scores),
                "alert": scores >= self.threshold,
            },
            index=features.index,
        )
        out["reasons"] = [[] for _ in range(len(out))]
        if explain and out["alert"].any():
            flagged = features[out["alert"].to_numpy()]
            explanations = self.explainer.explain(flagged)
            out.loc[flagged.index, "reasons"] = pd.Series(
                [e["reasons"] for e in explanations], index=flagged.index
            )
        return out

    def explain_one(self, features: pd.DataFrame) -> list[dict]:
        """Reasons for a single payment, whether or not it raised an alert."""
        return self.explainer.explain(features)[0]["reasons"]


def _card_profiles(transactions: pd.DataFrame) -> pd.DataFrame:
    """What an analyst would want to know about each card, from its genuine history."""
    genuine = transactions[transactions["is_fraud"] == 0]
    in_store = genuine[genuine["channel"] == "in_store"]
    usual_city = in_store.groupby("card_id")["merchant_city"].agg(lambda s: s.mode().iat[0])
    usual_device = genuine.groupby("card_id")["device_id"].agg(
        lambda s: s.dropna().mode().iat[0] if s.notna().any() else None
    )
    stats = transactions.groupby("card_id").agg(
        customer_id=("customer_id", "first"),
        payments=("transaction_id", "size"),
        avg_amount=("amount", "mean"),
        usual_hour=("timestamp", lambda s: int(s.dt.hour.median())),
        last_payment=("timestamp", "max"),
        frauds=("is_fraud", "sum"),
    )
    return stats.join(usual_city.rename("usual_city")).join(usual_device.rename("usual_device"))


def _ring_actors(
    groups: list[LinkedGroup], alerts: pd.DataFrame, transactions: pd.DataFrame
) -> dict[str, Actor]:
    """For each detected ring, the devices seen in its flagged payments and where they appear."""
    actors = {}
    for group in groups:
        if not group.is_ring:
            continue
        hits = alerts[alerts["card_id"].isin(group.cards) & alerts["device_id"].notna()]
        if hits.empty:
            continue
        rows = transactions.set_index("transaction_id").loc[hits["transaction_id"]]
        actors[group.group_id] = Actor(
            devices=tuple(sorted(hits["device_id"].unique())),
            lat=float(rows["lat"].median()),
            lon=float(rows["lon"].median()),
            ring_id=group.group_id,
        )
    return actors


def _warm_stores(transactions: pd.DataFrame, test_start: pd.Timestamp):
    """Replay history through the online engine: once up to the test period, once to the end."""
    store = OnlineFeatureStore()
    rows = iter_transactions(transactions)
    before = transactions["timestamp"] < test_start
    n_before = int(before.sum())
    for i, txn in enumerate(rows):
        if i == n_before:
            break
        store.update(txn)
    at_test_start = copy.deepcopy(store)
    for txn in iter_transactions(transactions[~before]):
        store.update(txn)
    return at_test_start, store


def build_demo_world(config: GeneratorConfig = APP_CONFIG) -> DemoWorld:
    customers, directory, _ = build_world(config)
    transactions = generate_transactions(config)
    features = build_features(transactions)
    # The demo world is small (about 13 attacks in its test period), so a validation-based
    # choice between the two models is close to a coin toss here. Serve the winner of the
    # full 60-day evaluation instead; the performance page says so openly.
    bundle = train(features, importance_repeats=3, seed=config.seed, serve=SERVED_MODEL)
    split = time_split(features)
    explainer = ReasonExplainer.from_bundle(bundle)

    def scored(part: pd.DataFrame) -> pd.DataFrame:
        cols = ["card_id", "amount", "is_fraud", "fraud_type", "timestamp"]
        return part[cols].assign(score=predict_scores(bundle["model"], part))

    valid_scored = scored(split.valid)
    test_scored = scored(split.test)
    genuine_valid = np.sort(valid_scored.loc[valid_scored["is_fraud"] == 0, "score"].to_numpy())

    test_start = split.test["timestamp"].min()
    test_transactions = transactions[transactions["timestamp"] >= test_start].reset_index(drop=True)
    store_at_test_start, store_full = _warm_stores(transactions, test_start)

    flagged = predict_scores(bundle["model"], features) >= bundle["threshold"]
    alerts = transactions.loc[flagged, ["transaction_id", "card_id", "timestamp", "device_id"]]
    groups, compromise = ring_report(transactions, alerts)

    return DemoWorld(
        config=config,
        customers=customers,
        directory=directory,
        transactions=transactions,
        features=features,
        split=split,
        bundle=bundle,
        explainer=explainer,
        genuine_valid_scores=genuine_valid,
        valid_scored=valid_scored,
        test_scored=test_scored,
        test_transactions=test_transactions,
        store_at_test_start=store_at_test_start,
        store_full=store_full,
        cards=_card_profiles(transactions),
        alerts=alerts,
        groups=groups,
        compromise=compromise,
        ring_actors=_ring_actors(groups, alerts, transactions),
    )


def features_frame(rows: list[dict]) -> pd.DataFrame:
    """Online-engine feature dicts -> a DataFrame in model column order."""
    return pd.DataFrame(rows)[FEATURE_COLUMNS]


def strip_labels(txn: dict) -> dict:
    return {k: v for k, v in txn.items() if k not in LABEL_COLUMNS}
