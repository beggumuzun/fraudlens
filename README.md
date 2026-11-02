# FraudLens

**Real-time, explainable card fraud detection.**
FraudLens scores every card payment the moment it happens, flags the suspicious ones and
explains *why* each one was flagged, so a fraud analyst can act on it in seconds.

![CI](https://github.com/beggumuzun/fraudlens/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

> Work in progress. See the [roadmap](#roadmap) for what is built so far.

---

## Why this project

Card fraud is rare (well under 1% of payments) but expensive, and a bank has only a few
milliseconds to decide whether to approve a payment. That makes it a hard engineering
problem, not just a modelling one:

- **Extreme class imbalance**: a model that says "never fraud" is 99.7% accurate and useless.
- **Time matters**: features such as "how many payments in the last 10 minutes" must be
  computed from the past only, otherwise the model cheats (data leakage).
- **Decisions must be explainable**: analysts and regulators need a reason, not just a score.
- **Latency budget**: scoring has to be fast enough to sit inside a payment flow.

## Architecture

```mermaid
flowchart LR
    G[Transaction generator] -->|live stream| API[FastAPI scoring service]
    API --> F[Online feature engine]
    F --> M[Fraud model]
    M --> X[Explainer - SHAP]
    API --> DB[(SQLite / PostgreSQL)]
    DB --> D[Analyst dashboard]
    G -->|historical data| T[Training pipeline]
    T --> M
```

## Roadmap

- [x] **1. Synthetic data engine**: customers with individual habits, UK and foreign merchants,
      three fraud attack types and genuine trips abroad as hard negatives
- [x] **2. Feature engineering**: 23 velocity, habit, location and device features,
      computed point-in-time by two engines (batch + online) that are tested to agree
- [x] **3. Modelling**: logistic regression baseline vs gradient boosting; time-based split;
      alert threshold set from a fixed false-alarm budget
- [ ] **4. Explainability**: SHAP reason codes for every flagged payment
- [ ] **5. Scoring API**: FastAPI service with measured p50 / p99 latency
- [ ] **6. Live demo**: replay stream + analyst dashboard
- [ ] **7. Production touches**: Docker, model versioning, demo GIF

## The data

Real fraud data is confidential, and public datasets either hide their features (PCA
components) or lack the behaviour patterns needed for real-time features. FraudLens therefore
generates its own data, built around attack patterns that banks actually see:

| Scenario | What happens | Signals a good model should pick up |
|---|---|---|
| `card_testing` | 5–12 tiny online payments seconds apart, then 1–2 big electronics purchases | burst of payments, new device, foreign location |
| `account_takeover` | 2–4 high-value online purchases between 01:00 and 05:00 from abroad | unusual hour, new device, amount far above habit |
| `stolen_card_spree` | 4–8 large in-store purchases in another UK city within a few hours | unusual city, amount far above habit, many shops quickly |
| `legit_travel` *(not fraud)* | a genuine holiday abroad with normal spending | foreign location, but normal hours and amounts |

`legit_travel` exists on purpose: without these hard negatives a model could simply flag
every foreign payment. The generator is fully reproducible: the same seed always gives the
same dataset, which is checked in the test suite.

Default dataset: **~130,000 transactions**, 2,000 customers, 30 days, **~0.3% fraud**.

### Schema

| Column | Description |
|---|---|
| `transaction_id` | unique id, in time order |
| `timestamp` | when the payment happened |
| `customer_id`, `card_id` | who paid |
| `merchant_id`, `merchant_category`, `merchant_city` | where they paid (`ONLINE` for e-commerce) |
| `channel` | `online` or `in_store` |
| `amount` | GBP |
| `lat`, `lon` | location of the payment |
| `device_id` | device used for online payments (empty in store) |
| `is_fraud`, `fraud_type` | labels: **never** used as model inputs |

## Features

Every feature for a payment is built **only from that card's earlier payments**. A test
deletes all future data, recomputes everything and checks that no past value changes.

| Group | Examples | Catches |
|---|---|---|
| Velocity | payments in last 10 min / 1 h / 24 h, small payments in last 10 min | card testing bursts |
| Habit | amount ÷ card's average, z-score, how usual this hour is, new category | spending out of character |
| Location | distance from usual place, distance and **travel speed** since last payment | stolen cards, "impossible travel" |
| Device | first use of a device, device first seen < 24 h ago | account takeover |

How strongly some of them separate fraud from genuine payments on the default dataset:

| Feature | Genuine (mean) | Fraud (mean) | Ratio |
|---|---:|---:|---:|
| `small_txn_count_10m` | 0.002 | 2.19 | ~976× |
| `is_outside_uk` | 0.004 | 0.68 | ~157× |
| `txn_count_10m` | 0.03 | 2.22 | ~80× |
| `dist_from_usual_km` | 55 | 3,506 | ~63× |
| `new_device_24h` | 0.03 | 0.68 | ~25× |
| `amount_to_avg_ratio` | 1.10 | 5.41 | ~5× |

No single feature is enough on its own (genuine customers also travel abroad and buy
expensive things), which is why a model that combines them is needed. That is Phase 3.

### Two engines, one definition

| | Batch engine | Online engine |
|---|---|---|
| Used for | building the training set | scoring a live payment |
| How | vectorised pandas over the whole history | small running state per card |
| Speed | ~130k payments in ~4 s | **p50 0.04 ms, p99 0.09 ms per payment** |

Running two implementations of the same logic is a well-known source of
*training/serving skew*. FraudLens guards against it with a parity test that replays a
whole dataset through the online engine and requires every feature to match the batch
engine. The test earned its place on day one: it caught a bug where the online engine
computed `~True` (which is `-2` in Python) instead of `False`.

## Model results

**Setup.** 5,000 customers over 60 days (651,546 payments, 0.14% fraud), split by time:
train on 1 Jan – 8 Feb, choose the alert threshold on 9 – 17 Feb, report on 18 Feb – 1 Mar.
The model never sees the future during training.

**The rule.** Analysts can only review so many alerts and every false alarm blocks a genuine
customer's card. So the threshold is set to allow **at most 1 false alarm per 1,000 genuine
payments** (chosen on the validation weeks), and models are compared at that fixed budget.

Test set (130,394 payments, 170 fraudulent, 24 separate attacks):

| Model | PR-AUC | Fraud payments caught | £ fraud caught | Attacks caught | Alert precision |
|---|---:|---:|---:|---:|---:|
| Logistic regression (baseline) | 0.571 | 81.2% | 86.7% | 23 / 24 | 45% |
| **Gradient boosting** | **0.969** | **98.2%** | **98.7%** | **24 / 24** | **55%** |

- Gradient boosting wins because fraud lives in *combinations* ("far from home **and** a new
  device **and** at night") that a linear model cannot express.
- Recall by attack type: account takeover 100%, card testing 100%, stolen card spree 94.9%.
- Most useful features (permutation importance): merchant category, distance from usual
  location, outside the UK, money spent in the last hour, amount z-score.
- Scoring one payment takes about **5 ms** (p50) on a laptop CPU.

Full numbers are in [`reports/metrics.json`](reports/metrics.json).

**Limitations, honestly.** The data is synthetic, so the attack patterns are cleaner than
real fraud, where criminals adapt to the model. The test set holds only 24 attacks, so
"24 / 24" is encouraging rather than proof. The value of this phase is the evaluation
method (time split, fixed false-alarm budget, money- and attack-level recall), which carries
over unchanged to real data.

## Quick start

```bash
git clone https://github.com/beggumuzun/fraudlens.git
cd fraudlens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. generate the dataset (the results above use exactly these settings)
fraudlens-generate --customers 5000 --days 60 --out data/transactions.csv

# 2. build features (also prints a fraud-vs-genuine report)
fraudlens-features --input data/transactions.csv --out data/features.csv

# 3. train and compare models (saves models/fraudlens.joblib and reports/metrics.json)
fraudlens-train --features data/features.csv

# run tests and lint
pytest -v
ruff check .
```

## Project structure

```
fraudlens/
├── src/fraudlens/
│   ├── data/
│   │   ├── entities.py    # cities, categories, customers, merchants
│   │   ├── scenarios.py   # fraud attacks and genuine travel
│   │   └── generator.py   # builds the dataset, CLI, live replay
│   ├── features/
│   │   ├── definitions.py # feature names, constants, shared maths
│   │   ├── batch.py       # vectorised engine for training + CLI
│   │   └── online.py      # per-payment engine for live scoring
│   └── model/
│       ├── split.py       # train / validation / test by time
│       ├── metrics.py     # false-alarm budget, £ recall, attack recall
│       └── train.py       # baseline vs gradient boosting, save model + CLI
├── reports/metrics.json   # latest training results
├── tests/                 # pytest suite, run on every push by GitHub Actions
└── pyproject.toml
```

## Author

**Begüm Uzun**, BSc Computer Science, UWE Bristol
[LinkedIn](https://www.linkedin.com/in/beggum-uzun) · [GitHub](https://github.com/beggumuzun)
