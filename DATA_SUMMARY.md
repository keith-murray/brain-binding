# Data Summary: Variable Binding through Episodic Memory

## Overview

Pilot fMRI experiment testing relational pattern completion (ABA vs. ABB) using 4 visual stimuli (circle, rectangle, star, triangle). Each session includes a working-memory localizer block followed by 4 runs of the main relational task.

---

## Directory Structure

```
data/
  <YYYY-MM-DD_HH-MM-SS>/        # session timestamp
    <UUID>.csv                   # trial-by-trial behavioral data (1 row/trial)
    <UUID>.log                   # PsychoPy event log
```

### Sessions

| Folder | UUID | Trials (rows incl. header) |
|---|---|---|
| `2026-03-02_15-10-18` | `cc2bdacf-6fe4-4e56-9630-a6ab21438244` | 144 trials |
| `2026-03-02_16-25-49` | `7ea7ce2b-65c3-4ff4-a9bd-ded1149f0b99` | 144 trials |
| `2026-03-04_18-13-59` | `e3f4c2c2-e078-42a1-970c-cc7833d2a7b2` | 144 trials |

Each session has **144 trials total**: 24 WM + 120 main task (4 runs × 30 trials).

---

## Task Structure

### Block 1 — Working Memory (WM) Localizer
- **24 trials** run once at session start
- **Design:** 4 shapes × 2 ISI conditions × 3 repetitions
- **Trial:** fixation → sample stimulus (0.5 s) → delay (ISI) → 4AFC test display → response (2 s deadline)
- **Purpose:** working memory control localizer

### Blocks 2–5 — Main Relational Task (`main_run1` … `main_run4`)
- **30 trials per run, 120 total**
- **Rule types:** ABA (e.g., circle→triangle→circle) or ABB (e.g., circle→triangle→triangle)
- **Trial:** fixation → rule sequence (3 stimuli, ~1 s each with jittered ISIs) → test sequence (2 stimuli) → 4AFC response display → response → ITI
- **Purpose:** relational pattern completion; 4AFC response mapping randomized each trial to reduce motor confounds

---

## CSV Columns

### Shared Columns (all blocks)

| Column | Type | Description |
|---|---|---|
| `block` | string | `"WM"`, `"main_run1"`, `"main_run2"`, `"main_run3"`, `"main_run4"` |
| `sid` | UUID string | Session/participant identifier |
| `trial` | int | Trial index, continuous across blocks within a session |
| `slot_mapping` | string | 4-item comma-separated list of shapes in response slot order (e.g., `"star,rectangle,circle,triangle"`) |
| `correct_stim` | string | The correct shape to select |
| `response_key` | int | Key pressed (1–4) |
| `response_slot` | int | Slot selected (1–4) |
| `response_stim` | string | Shape at the selected slot |
| `correct` | int | 1 = correct, 0 = incorrect |
| `rt` | float | Reaction time in seconds from test display onset (empty if no response) |
| `t_fixation` | float | Absolute time (s from session start) of fixation onset |
| `t_response` | float | Absolute time of response |

---

### WM Block Only (`block == "WM"`)

| Column | Type | Description |
|---|---|---|
| `sample_stim` | string | Shape shown during sample phase (`circle`, `rectangle`, `star`, `triangle`) |
| `isi_condition` | float | Delay duration: `1.0` or `2.5` seconds |
| `t_sample` | float | Absolute time of sample stimulus onset |
| `t_delay` | float | Absolute time of delay (ISI) onset |

*Unused in WM:* `rule_type`, `A_stim`, `B_stim`, `A_prime`, `B_prime`, `rule_sequence`, `test_sequence`, `correct_next_stim`, `isi1`–`isi5`, `iti`, `t_rule1`–`t_rule3`, `t_test1`, `t_test2`

---

### Main Task Only (`block == "main_runN"`)

| Column | Type | Description |
|---|---|---|
| `rule_type` | string | `"ABA"` or `"ABB"` |
| `A_stim` | string | Shape assigned to role A in rule sequence |
| `B_stim` | string | Shape assigned to role B in rule sequence |
| `A_prime` | string | New shape assigned to role A in test sequence |
| `B_prime` | string | New shape assigned to role B in test sequence |
| `rule_sequence` | string | 3-item comma-separated sequence shown during rule phase (e.g., `"circle,triangle,circle"`) |
| `test_sequence` | string | 2-item comma-separated sequence shown during test phase (e.g., `"triangle,circle"`) |
| `correct_next_stim` | string | Correct third item completing the test sequence |
| `isi1`–`isi5` | float | Jittered ISI durations (s) between stimuli; drawn from truncated Normal(μ=1.0, σ=0.2, min=0) |
| `iti` | float | Inter-trial interval (s); drawn from truncated Normal(μ=3.0, σ=1.0, min=0) |
| `t_rule1` | float | Absolute time of first rule stimulus onset |
| `t_rule2` | float | Absolute time of second rule stimulus onset |
| `t_rule3` | float | Absolute time of third rule stimulus onset |
| `t_test1` | float | Absolute time of first test stimulus onset |
| `t_test2` | float | Absolute time of second test stimulus onset |

*Unused in main task:* `sample_stim`, `isi_condition`, `t_sample`, `t_delay`

---

## Stimuli

Four shapes (`assets/`): `circle.png`, `rectangle.png`, `star.png`, `triangle.png`

---

## Log File Format

PsychoPy plain-text log, one entry per line:

```
<elapsed_seconds>    <LEVEL>    <message>
```

Key `DATA` entries:
- `Keypress: equal` — scanner TTL pulse (TR trigger)
- `Keypress: <1–4>` — participant response
- `WM trial N: sample=<shape>, isi=<duration>` — WM trial start
- `WM trial N response: key=<k>, rt=<rt>, correct=<0|1>` — WM trial response
