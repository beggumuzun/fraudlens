# Model results

**Setup.** 5,000 customers over 60 days (651,379 payments, 0.15% fraud, 8 fraud rings), split by time:
train on 1 Jan – 8 Feb, choose the alert threshold on 9 – 17 Feb, report on 18 Feb – 1 Mar.
The model never sees the future during training.

**The rule.** Analysts can only review so many alerts and every false alarm blocks a genuine
customer's card. So the threshold is set to allow **at most 1 false alarm per 1,000 genuine
payments** (chosen on the validation weeks), and models are compared at that fixed budget.

Test set (130,510 payments, 173 fraudulent, 29 separate attacks):

| Model | PR-AUC | Fraud payments caught | £ fraud caught | Attacks caught | Alert precision |
|---|---:|---:|---:|---:|---:|
| Logistic regression (baseline) | 0.850 | 91.9% | 90.4% | 29 / 29 | 49% |
| **Gradient boosting** | **0.970** | **98.3%** | **98.7%** | **29 / 29** | **55%** |

- Gradient boosting wins because fraud lives in *combinations* ("far from home **and** a new
  device **and** at night") that a linear model cannot express. The gap is largest on
  stolen-card sprees (96% caught vs 84%) and in how clean the alerts are (PR-AUC 0.97 vs 0.85).
- Recall by attack type: account takeover 100%, card testing 100%, stolen card spree 95.7%.
- Most useful features (permutation importance): distance from usual location, device first
  seen in the last 24 h, merchant category, amount vs habit, money spent in the last hour.
- Scoring one payment takes about **7 ms** (p50, `fraudlens-bench`, 2-core cloud machine).

## A hypothesis I got wrong

After fraud rings and shared family devices were added to
the data, the baseline collapsed from 0.67 to 0.29 PR-AUC while gradient boosting held
steady. My first guess was that "several cards on one device" confused it, being
suspicious for a ring but normal for a family. An ablation disproved that:

| Baseline variant | PR-AUC |
|---|---:|
| As it was | 0.29 |
| Without the shared-device feature | 0.26 |
| **With log-transformed skewed features** | **0.85** |

The real cause was preprocessing. Counts, amounts and distances are heavily skewed (2 km
or 5,000 km), and a linear model, with one weight per feature, gets dominated by the
extremes. Applying `log1p` before scaling is standard practice for linear models, so
the baseline now does it. Gradient boosting is unaffected (trees only care about the
order of values) and still wins, but now against a fair opponent.

**Which model the app serves.** On the app's much smaller demo world (2,000 customers,
45 days, 13 attacks in its test weeks) the baseline actually comes out slightly ahead:
0.935 vs 0.932 PR-AUC on validation, 0.906 vs 0.847 on test, and 13 of 13 attacks caught
against 12 of 13. With so few attacks, one attack is worth 8 percentage points, so these
gaps are noise in either direction. The 60-day evaluation above (29 attacks, 130,510 test
payments) is the more reliable guide, and there gradient boosting wins clearly. The app
therefore serves gradient boosting, and its performance screen shows both models' demo
numbers and says why.

Full numbers are in [`reports/metrics.json`](../reports/metrics.json).

**Limitations, honestly.** The data is synthetic, so the attack patterns are cleaner than
real fraud, where criminals adapt to the model. The test set holds only 29 attacks, so
"29 / 29" is encouraging rather than proof. The value of this phase is the evaluation
method (time split, fixed false-alarm budget, money- and attack-level recall), which carries
over unchanged to real data.

---

[Back to the README](../README.md)
