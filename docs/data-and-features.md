# Data and features

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
| fraud rings | 60% of online attacks come from rings: one criminal group attacking about 8 cards from the same 1-3 devices, after buying details stolen from one breached shop that every victim paid at days earlier | new device per payment; across cards, the shared devices and the common shop |
| shared family devices *(not fraud)* | 4% of customers pay online from a laptop or tablet shared with one or two family members | a shared device that is perfectly innocent |

`legit_travel` and shared family devices exist on purpose: without these hard negatives a
model could simply flag every foreign payment, or every device used by more than one card. The generator is fully reproducible: the same seed always gives the
same dataset, which is checked in the test suite.

Default dataset: **~130,000 transactions**, 2,000 customers, 30 days, **~0.3% fraud**,
3 fraud rings.

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
| `is_fraud`, `fraud_type`, `ring_id` | labels: **never** used as model inputs |
| `breach_ring_id` | answer key: marks the genuine payment where a ring stole the card's details, used only to check the ring analysis |

## Features

Every feature for a payment is built **only from that card's earlier payments**. A test
deletes all future data, recomputes everything and checks that no past value changes.

| Group | Examples | Catches |
|---|---|---|
| Velocity | payments in last 10 min / 1 h / 24 h, small payments in last 10 min | card testing bursts |
| Habit | amount ÷ card's average, z-score, how usual this hour is, new category | spending out of character |
| Location | distance from usual place, distance and **travel speed** since last payment | stolen cards, "impossible travel" |
| Device | first use of a device, device first seen < 24 h ago | account takeover |
| Network | how many *other* cards used this device before | meant for fraud rings, but adds nothing measurable (see [the ablation](fraud-rings.md#a-second-hypothesis-i-got-wrong)) |

How strongly some of them separate fraud from genuine payments on the default dataset:

| Feature | Genuine (mean) | Fraud (mean) | Ratio |
|---|---:|---:|---:|
| `small_txn_count_10m` | 0.002 | 1.81 | ~770× |
| `is_outside_uk` | 0.005 | 0.61 | ~120× |
| `txn_count_10m` | 0.03 | 1.84 | ~65× |
| `device_card_count` | 0.02 | 1.25 | ~58× |
| `dist_from_usual_km` | 54 | 1,867 | ~35× |
| `new_device_24h` | 0.03 | 0.67 | ~25× |
| `amount_to_avg_ratio` | 1.10 | 4.70 | ~4× |

No single feature is enough on its own (genuine customers also travel abroad and buy
expensive things), which is why a model that combines them is needed: see [Model results](model.md).

### Two engines, one definition

| | Batch engine | Online engine |
|---|---|---|
| Used for | building the training set | scoring a live payment |
| How | vectorised pandas over the whole history | small running state per card |
| Speed | ~130k payments in ~4 s | **p50 0.06 ms, p99 0.14 ms per payment** |

Running two implementations of the same logic is a well-known source of
*training/serving skew*. FraudLens guards against it with a parity test that replays a
whole dataset through the online engine and requires every feature to match the batch
engine. The test earned its place on day one: it caught a bug where the online engine
computed `~True` (which is `-2` in Python) instead of `False`.

Generated data is well-behaved, so hand-made cases cover what it never produces, each
run through both engines:

- **Ties.** Two payments at the same second are ordered by `transaction_id` in both engines.
- **Late payments.** A payment that arrives older than the latest one already recorded,
  on any card, is rejected (`OutOfOrderError`) instead of silently producing wrong
  features. The check is global because shared devices link cards: a late payment on one
  card would change another card's device count after the fact.
- **Online payments with no device ID** count as one unknown device per card in both
  engines, kept separate from in-store payments, so the first one is always "new".
- **Rounding noise.** Five payments of exactly £0.10 have no spread, so a £0.11 payment
  gets a z-score of 0. The old formula's float cancellation gave it +50.

Latency figures come from `fraudlens-bench` ([`reports/benchmark.json`](../reports/benchmark.json)),
run on a 2-core cloud machine. The quick timing inside `reports/metrics.json` is taken
during training, alongside other work, and is noisier.

---

[Back to the README](../README.md)
