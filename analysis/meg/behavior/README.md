# Behavioral analysis for MEG data

## Summary statistics (sub-001)

### Panel (0,0) — Overall accuracy

| Metric | Value |
|---|---|
| Accuracy | 115/120 = 95.83% |
| Binomial test vs chance (0.5) | p < 0.0001 |

### Panel (0,1) — Accuracy by rule type

| Condition | Accuracy | Binomial p |
|---|---|---|
| ABA | 59/60 = 98.33% | < 0.0001 |
| ABB | 56/60 = 93.33% | < 0.0001 |

| Test | Result |
|---|---|
| Chi-square (ABA vs ABB) | χ²(1) = 0.8348, p = 0.3609 |

### Panel (0,2) — Accuracy by trial type

| Condition | Accuracy | Binomial p |
|---|---|---|
| Match | 56/60 = 93.33% | < 0.0001 |
| Rule order | 30/30 = 100.00% | < 0.0001 |
| Random | 29/30 = 96.67% | < 0.0001 |

| Test | Result |
|---|---|
| Chi-square (across trial types) | χ²(2) = 2.2957, p = 0.3173 |
| Fisher exact (match vs rule_order) | p = 0.2969 |
| Fisher exact (match vs random) | p = 0.6613 |
| Fisher exact (rule_order vs random) | p = 1.0000 |

### Panel (1,0) — RT distribution per participant (correct trials)

| Participant | Median RT (s) |
|---|---|
| S1 | 0.3655 |

### Panel (1,1) — RT by rule type (correct trials, aggregated)

| Condition | n | Median RT (s) |
|---|---|---|
| ABA | 59 | 0.3175 |
| ABB | 56 | 0.4239 |

| Test | Result |
|---|---|
| MWU (ABA vs ABB) | U = 1305.0, p = 0.0525 |

### Panel (1,2) — RT by trial type (correct trials, aggregated)

| Condition | n | Median RT (s) |
|---|---|---|
| Match | 56 | 0.4007 |
| Rule order | 30 | 0.4205 |
| Random | 29 | 0.2890 |

| Test | Result |
|---|---|
| MWU (match vs rule_order) | U = 853.0, p = 0.9098 |
| MWU (match vs random) | U = 1056.0, p = 0.0240 |
| MWU (rule_order vs random) | U = 580.0, p = 0.0285 |
