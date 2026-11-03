# FraudLens

**Real-time, explainable card fraud detection.**
FraudLens scores every card payment the moment it happens, flags the suspicious ones and
explains *why* each one was flagged, so a fraud analyst can act on it in seconds.

![CI](https://github.com/beggumuzun/fraudlens/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

> Work in progress. See the [roadmap](#roadmap) for what is built so far.

---

## The app

An analyst app (Streamlit) puts the whole system on screen. It needs no data or model
files: on first launch it generates its own bank of 2,000 customers, trains the model and
warms up the live feature engine, which takes about 30 seconds.

![Live monitor](docs/screenshots/live-monitor.png)

| Screen | What it shows |
|---|---|
| **Live monitor** | Unseen payments replayed in time order through the real-time path (online features, model, explanations). A "fraud seismograph" draws every payment as a line whose height is its fraud score on a log scale, so attacks punch through the alert line as red spikes. Alerts queue up with their reasons, next to the ground truth. |
| **Test a payment** | Pick a real customer, describe a new payment (amount, place, time, device) and get the decision and its reasons instantly, from that card's actual history. |
| **Attack simulator** | Launch a fresh card-testing, account-takeover or stolen-card attack on any customer and see at which payment it is caught and how much money is saved. |
| **Model performance** | Baseline vs gradient boosting, and a slider for the false-alarm budget showing how many frauds, pounds and attacks each setting catches. |

<details>
<summary>More screenshots</summary>

**Test a payment**: a night-time jewellery purchase from Lagos on an unknown device
![Test a payment](docs/screenshots/test-a-payment.png)

**Attack simulator**: a stolen-card spree, caught on the third payment
![Attack simulator](docs/screenshots/attack-simulator.png)

**Model performance**
![Model performance](docs/screenshots/model-performance.png)

</details>

Risk is shown on a 0-100 scale: the share of genuine payments that look less suspicious.
Raw fraud probabilities are tiny because fraud is rare, so "riskier than 99.95% of genuine
payments" is easier to act on than "score 0.004". Alerts start at 99.9.

```bash
fraudlens-app        # opens http://localhost:8501
```

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
    G[Synthetic data engine] -->|history| B[Batch feature engine]
    B --> T[Training: baseline vs gradient boosting]
    T --> M[Fraud model + alert threshold]
    G -->|live payments| O[Online feature engine]
    O --> M
    M --> X[Explainer: group Shapley reasons]
    X --> A[Analyst app]
```

## Roadmap

- [x] **1. Synthetic data engine**: customers with individual habits, UK and foreign merchants,
      three fraud attack types and genuine trips abroad as hard negatives
- [x] **2. Feature engineering**: 23 velocity, habit, location and device features,
      computed point-in-time by two engines (batch + online) that are tested to agree
- [x] **3. Modelling**: logistic regression baseline vs gradient boosting; time-based split;
      alert threshold set from a fixed false-alarm budget
- [x] **4. Explainability**: plain-English reason codes from exact Shapley values over
      feature groups; used to find and fix a shortcut the model had learned
- [x] **5. Analyst app**: live monitor, payment tester, attack simulator and model
      performance screens, tested headlessly with Streamlit's AppTest
- [ ] **6. Scoring API**: FastAPI service with measured p50 / p99 latency
- [ ] **7. Production touches**: public deployment, Docker, model versioning, demo GIF

## The data

Real fraud data is confidential, and public datasets either hide their features (PCA
components) or lack the behaviour patterns needed for real-time features. FraudLens therefore
generates its own data, built around attack patterns that banks actually see:

| Scenario | What happens | Signals a good model should pick up |
|---|---|---|
| `card_testing` | 5–12 tiny online payments seconds apart, then 1–2 big electronics purchases; half appear to be in the UK (VPN) | burst of payments, new device |
| `account_takeover` | 2–4 high-value online purchases between 01:00 and 05:00, mostly from abroad (30% via a UK VPN) | new device, unusual hour, amount far above habit |
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
| `small_txn_count_10m` | 0.002 | 2.09 | ~929× |
| `is_outside_uk` | 0.005 | 0.41 | ~80× |
| `txn_count_10m` | 0.03 | 2.12 | ~76× |
| `dist_from_usual_km` | 51 | 2,281 | ~45× |
| `new_device_24h` | 0.03 | 0.69 | ~25× |
| `amount_to_avg_ratio` | 1.10 | 4.50 | ~4× |

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

**Setup.** 5,000 customers over 60 days (651,690 payments, 0.15% fraud), split by time:
train on 1 Jan – 8 Feb, choose the alert threshold on 9 – 17 Feb, report on 18 Feb – 1 Mar.
The model never sees the future during training.

**The rule.** Analysts can only review so many alerts and every false alarm blocks a genuine
customer's card. So the threshold is set to allow **at most 1 false alarm per 1,000 genuine
payments** (chosen on the validation weeks), and models are compared at that fixed budget.

Test set (130,408 payments, 196 fraudulent, 28 separate attacks):

| Model | PR-AUC | Fraud payments caught | £ fraud caught | Attacks caught | Alert precision |
|---|---:|---:|---:|---:|---:|
| Logistic regression (baseline) | 0.670 | 72.4% | 83.3% | 24 / 28 | 45% |
| **Gradient boosting** | **0.957** | **97.4%** | **98.7%** | **28 / 28** | **61%** |

- Gradient boosting wins because fraud lives in *combinations* ("far from home **and** a new
  device **and** at night") that a linear model cannot express. The gap is largest on
  stolen-card sprees: 92% caught vs 38% for the baseline.
- Recall by attack type: account takeover 100%, card testing 100%, stolen card spree 92.3%.
- Most useful features (permutation importance): distance from usual location, device first
  seen in the last 24 h, money spent in the last hour, merchant category, amount vs habit.
- Scoring one payment takes about **5 ms** (p50) on a laptop CPU.

Full numbers are in [`reports/metrics.json`](reports/metrics.json).

**Limitations, honestly.** The data is synthetic, so the attack patterns are cleaner than
real fraud, where criminals adapt to the model. The test set holds only 28 attacks, so
"28 / 28" is encouraging rather than proof. The value of this phase is the evaluation
method (time split, fixed false-alarm budget, money- and attack-level recall), which carries
over unchanged to real data.

## Explainability

Every alert comes with plain-English reasons, ranked by how much of the suspicion each
one carries. A real example from the test period:

```
T0542864  account takeover  score 0.994
   51%  Online payment from a device never used with this card before
   17%  £899.94 is 9.8x this card's average payment
   17%  Paid outside the UK, 5,037 km from the card's usual area
```

**How.** Features are grouped by meaning (location, velocity, amount, device, timing,
merchant, history). The payment is re-scored for all 2⁷ = 128 mixes of "this group has
its real value" and "this group looks like a typical genuine payment", and each group
receives its exact **Shapley value**: its fair share of the score. Explaining one
payment takes about 9 ms.

Why not simply switch off one signal at a time? Fraud is an *AND* of signals. Making
either "new device" or "abroad" look normal drops the score to almost nothing, so both
would look 99% responsible. Shapley values split the credit fairly, and a unit test
checks exactly this on a model whose answer is known by hand.

**What the explanations revealed.** The first version of the explainer showed that
every card-testing alert was explained by *"paid from abroad + digital goods"*, not by
the burst of tiny payments. The model had learned a shortcut: in the generated data,
card testers were always abroad, so it would have missed any tester using a UK VPN.
The generator was fixed (half of card testers and 30% of account takeovers now appear
in the UK), the model retrained, and the reasons now match the attacks:

| Alert type | Top reason: device | velocity | location | amount |
|---|---:|---:|---:|---:|
| card testing | **108** | 6 | 0 | 0 |
| account takeover | **17** | 0 | 0 | 0 |
| stolen card spree | 0 | 11 | **48** | 1 |
| false alarm (genuine) | 0 | 7 | **97** | 18 |

The last row is useful too: most false alarms are genuine customers spending more
than usual in another UK city, which looks exactly like a stolen-card spree. That
points to the next improvement: features that know about a customer's travel history.

Example explanations are in [`reports/explanations.json`](reports/explanations.json).

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

# 4. explain the alerts on the test period (writes reports/explanations.json)
fraudlens-explain --features data/features.csv

# 5. open the analyst app (builds its own demo world, no files needed)
fraudlens-app

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
│   ├── model/
│   │   ├── split.py       # train / validation / test by time
│   │   ├── metrics.py     # false-alarm budget, £ recall, attack recall
│   │   └── train.py       # baseline vs gradient boosting, save model + CLI
│   ├── explain/
│   │   ├── reasons.py     # group Shapley values -> plain-English reasons
│   │   └── cli.py         # explain test-period alerts
│   └── app/
│       ├── world.py       # demo world: data, model, warmed-up live engine
│       ├── main.py        # navigation and layout
│       └── views/         # live monitor, test a payment, simulator, performance
├── streamlit_app.py       # entry point for Streamlit Community Cloud
├── docs/screenshots/      # images used in this README
├── reports/               # latest training metrics and example explanations
├── tests/                 # pytest suite, run on every push by GitHub Actions
└── pyproject.toml
```

## Author

**Begüm Uzun**, BSc Computer Science, UWE Bristol
[LinkedIn](https://www.linkedin.com/in/beggum-uzun) · [GitHub](https://github.com/beggumuzun)
