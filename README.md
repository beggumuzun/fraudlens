# FraudLens

**Real-time, explainable card fraud detection.**
FraudLens scores every card payment the moment it happens, flags the suspicious ones and
explains *why* each one was flagged, so a fraud analyst can act on it in seconds.

![CI](https://github.com/beggumuzun/fraudlens/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

> 🚧 Work in progress. See the [roadmap](#roadmap) for what is built so far.

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
- [ ] **2. Feature engineering**: velocity, spend-vs-habit, location and device features,
      computed point-in-time with no leakage
- [ ] **3. Modelling**: logistic regression baseline → gradient boosting; time-based split;
      PR-AUC and recall at a fixed false-alarm budget
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

## Quick start

```bash
git clone https://github.com/beggumuzun/fraudlens.git
cd fraudlens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# generate the dataset
fraudlens-generate --customers 2000 --days 30 --out data/transactions.csv

# run tests and lint
pytest -v
ruff check .
```

## Project structure

```
fraudlens/
├── src/fraudlens/
│   └── data/
│       ├── entities.py    # cities, categories, customers, merchants
│       ├── scenarios.py   # fraud attacks and genuine travel
│       └── generator.py   # builds the dataset, CLI, live replay
├── tests/                 # pytest suite, run on every push by GitHub Actions
└── pyproject.toml
```

## Author

**Begüm Uzun**, BSc Computer Science, UWE Bristol
[LinkedIn](https://www.linkedin.com/in/beggum-uzun) · [GitHub](https://github.com/beggumuzun)
