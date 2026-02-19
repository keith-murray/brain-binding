# Variable Binding through Episodic Memory

## Relational Reasoning (`relational/run.py`)

A pilot fMRI task testing ABA vs. ABB relational pattern completion using 4 visual stimuli (circle, rectangle, star, triangle). Each session includes:

1. **Working memory control** (24 trials) — see a shape, maintain it across a short or long delay, then select it from a 4AFC display.
2. **Rest** (2 min, fixation).
3. **Main relational task** (4 runs × 30 trials) — observe a 3-item rule sequence (ABA or ABB), then complete a 2-item test sequence via 4AFC. Response mapping is randomized each trial to reduce motor confounds.

Data are saved to `data/<timestamp>/` as a crash-safe CSV with per-trial onset times, jittered ISIs, response keys, and accuracy. Run with:
```
uv run python relational/run.py
```
