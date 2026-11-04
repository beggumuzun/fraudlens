# Explainability

Every alert comes with plain-English reasons, ranked by how much of the suspicion each
one carries. A real example from the test period:

```
T0553371  account takeover by a fraud ring  score 0.999
   42%  Online payment from a device never used with this card before
   25%  Paid 258 km from the card's usual area
   18%  Merchant category: digital goods
```

**How.** Features are grouped by meaning (location, velocity, amount, device, network,
timing, merchant, history). The payment is re-scored for all 2⁸ = 256 mixes of "this
group has its real value" and "this group looks like a typical genuine payment", and
each group receives its exact **Shapley value**: its fair share of the score. Explaining
one payment takes about 15 ms (p50, `fraudlens-bench`).

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
| card testing | 32 | **40** | 0 | 0 |
| account takeover | **30** | 0 | 1 | 0 |
| stolen card spree | 0 | 10 | **57** | 0 |
| false alarm (genuine) | 0 | 6 | **128** | 3 |

The last row is useful too: most false alarms are genuine customers spending more
than usual in another UK city, which looks exactly like a stolen-card spree. That
points to the next improvement: features that know about a customer's travel history.

**A sentence that was not true.** The example above used to read *"42% This device has
already been used with 6 other cards"*. The shared-device count lived in the device
group, and the sentence was chosen by the feature's value, not by its contribution.
Every ring payment is *also* from a brand-new device, and that is what the model
actually reacted to: setting the shared-device count to zero leaves the score unchanged
at 0.9985. The count now has its own "network" group, so it only gets a sentence when it
really carries at least 10% of the score. Here it carries 0.3%. A unit test checks
the situation with a model whose answer is known by hand: a payment that is both from a
new device and from a shared one, scored by a model that only reacts to the new device.

Example explanations are in [`reports/explanations.json`](../reports/explanations.json).

---

[Back to the README](../README.md)
