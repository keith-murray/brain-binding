# Variable Binding through Episodic Memory

## fMRI Relational Reasoning Experiment (`relational/run_fmri.py`)

A pilot fMRI task testing ABA vs. ABB relational pattern completion using 4 visual stimuli (circle, rectangle, star, triangle). Each session includes:

1. **Working memory control** (24 trials) — see a shape, maintain it across a short or long delay, then select it from a 4AFC display.
2. **Rest** (2 min, fixation).
3. **Main relational task** (4 runs × 30 trials) — observe a 3-item rule sequence (ABA or ABB), then complete a 2-item test sequence via 4AFC. Response mapping is randomized each trial to reduce motor confounds.

Data are saved to `data/<timestamp>/` as a crash-safe CSV with per-trial onset times, jittered ISIs, response keys, and accuracy. Run with:
```
uv run python relational/run_fmri.py
```

## MEG Relational Reasoning Experiment (`relational/run_meg.py`)

An MEG port of the relational reasoning task. The 4AFC response and working memory block have been removed; the paradigm is redesigned for MEG acquisition constraints.

**Paradigm (4 runs × 30 trials = 120 trials):**

Each trial presents two sequences of 3 shapes. The subject judges whether the test sequence follows the same abstract rule (ABA or ABB) as the rule sequence.

- **Match (50%)** — test sequence follows the same rule using new shapes.
- **Mismatch — rule order (25%)** — same test shapes, opposite rule (ABA→ABB or ABB→ABA).
- **Mismatch — random (25%)** — 3rd test shape drawn randomly from shapes not in the test pair.

Response: `1` = match, `2` = no match (2 s deadline).

**Trial structure:**

| Event | Duration | Fixation color |
|---|---|---|
| Fixation | 0.5 s | Black |
| Rule stims 1–3 | 0.5 s each, ~1 s ISI | — |
| Transition cue | 1.5 s fixed | Yellow |
| Test stims 1–3 | 0.5 s each, ~1 s ISI | — |
| Response window | up to 2 s | Black |
| Feedback | 0.5 s | Green (correct) / Red (incorrect) |
| Blink window | 1.5 s | Blue |
| ITI | ~3 s | Black |

**Triggers (VPIXX Pixel Mode, blue channel, values 0–7):**

| Code | Event |
|---|---|
| 0 | Reset |
| 1 | Fixation onset |
| 2 | Rule stim 1 |
| 3 | Rule stim 2 |
| 4 | Rule stim 3 |
| 5 | Test stim 1 |
| 6 | Test stim 2 |
| 7 | Test stim 3 |

Data are saved to `data/sub-<id>_ses-<n>_task-relational/` as a BIDS-style CSV and log file. Run with:
```
uv run python relational/run_meg.py
```
