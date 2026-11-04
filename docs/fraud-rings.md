# Fraud rings

Criminals rarely attack one card. A ring buys a batch of card details, usually stolen
from one breached shop, and works through them from a handful of devices. FraudLens
looks for that pattern in three steps.

**1. Link cards through devices.** Every online payment links a card to a device. A
**union-find** (disjoint-set) structure, written from scratch with path compression and
union by size, joins these links into groups of cards connected directly or through a
chain of shared devices. It is checked against breadth-first search on random graphs.

**2. Tell rings from families.** Structure alone is not enough: three people sharing a
family laptop form exactly the same shape as a small ring. A group is called a ring only
when the model has flagged at least two of its cards. The graph and the model each catch
what the other misses.

**3. Find the leak.** For each ring, which shop did the victims pay at in the 30 days
before their first alert, far more often than customers in general? That shop is the
likely point of compromise.

Results on the 60-day dataset (8 real rings, 62 victim cards, 35 linked groups). The
headline column uses only alerts the model was **not** trained on (validation and test
weeks). On its training weeks the model has seen the answers, so the retrospective
column is an upper bound, shown for comparison:

| | Out of sample | Retrospective (all alerts) |
|---|---|---|
| Rings found | **7 of 8** | 8 of 8 |
| Victim cards recovered | **55 of 62** | 62 of 62 |
| Family groups wrongly called a ring | **0 of 27** | 1 of 27 |
| Breached shop ranked first | **7 of 7** rings found | 8 of 8 |

The breached shop is checked against the generator's answer key (`breach_ring_id`), and
a test asserts it. Out of sample, every flagged card of every ring had paid at that shop
in the 30 days before its first alert, **3.4x to 9.9x** more often than customers
overall. The missed ring struck only one of its victims after the training weeks,
and one flagged card is not enough: a group needs two before it is called a ring.

The alerts are used exactly as an analyst would get them, mistakes included. Of the 27
victim cards flagged out of sample, 3 were flagged only on genuine payments: their
attacks had ended in the training weeks, and a later false alarm happened to hit them.
They still help link their ring. Real alerts are just as mixed, which is why a ring
needs at least two flagged cards and is confirmed by the shared shop, not by one alert.

## A second hypothesis I got wrong

The network feature (how many other cards used this
device before) looked like the perfect real-time ring signal: 53x higher on fraud than
on genuine payments in the 60-day data. An ablation (`fraudlens-ablation`) retrained the
model without it:

| Test set, gradient boosting | With the feature | Without it |
|---|---:|---:|
| PR-AUC | 0.970 | 0.970 |
| Fraud payments caught | 98.3% | 98.3% |
| Ring attacks caught on their first payment | 11 of 11 | 11 of 11 |

It adds nothing. Ring devices are always new to the victim's card, and the model already
catches new devices, so the extra signal is redundant on this data. The feature stays in
because it costs almost nothing, but the README no longer claims it catches rings. The rings are exposed by the graph, not by
this feature. Full numbers are in [`reports/ablation.json`](../reports/ablation.json).

Honest caveats: a ring's group can also pull in a victim's family members, who share
the victim's own device; the app shows them as linked but not flagged. And in real data
rings hide behind fresh devices, so production systems link cards through many more
signals (IP addresses, shipping addresses, emails) than this synthetic one does.

Full report: [`reports/rings.json`](../reports/rings.json).

---

[Back to the README](../README.md)
