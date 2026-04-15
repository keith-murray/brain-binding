# Data Format

## Directory Structure

```
data/
  <YYYY-MM-DD_HH-MM-SS>/       # session timestamp (folder created at run start)
    <UUID>.csv                  # trial-by-trial behavioral data
    <UUID>.log                  # PsychoPy event log
```

Each session folder contains exactly two files sharing the same UUID, which serves as the participant/session identifier (`sid`).

---

## CSV File

One row per trial. All trials from the same session (WM block + all main runs) are written to the same file in order.

### Header

```
block, sid, trial,
sample_stim, isi_condition,
rule_type, A_stim, B_stim, A_prime, B_prime, rule_sequence, test_sequence,
correct_next_stim, slot_mapping, correct_stim,
response_key, response_slot, response_stim, correct, rt,
isi1, isi2, isi3, isi4, isi5, iti,
t_fixation, t_sample, t_delay,
t_rule1, t_rule2, t_rule3, t_test1, t_test2, t_response
```

### Shared Columns (all blocks)

| Column | Type | Description |
|---|---|---|
| `block` | string | `"WM"` or `"main_run1"`, `"main_run2"`, etc. |
| `sid` | UUID string | Session/participant identifier |
| `trial` | int | Trial index, continuous across blocks within a session |
| `slot_mapping` | string | Comma-separated 4-item list of shapes in response slot order (e.g. `"star,rectangle,circle,triangle"`) |
| `correct_stim` | string | The correct shape to select |
| `response_key` | int | Key pressed (1–4) |
| `response_slot` | int | Slot selected (1–4) |
| `response_stim` | string | Shape at the selected slot |
| `correct` | int | 1 = correct, 0 = incorrect |
| `rt` | float | Reaction time in seconds (from test display onset); empty if no response |
| `t_fixation` | float | Absolute time (s from session start) of fixation cross onset |
| `t_response` | float | Absolute time of response |

---

### WM Block (`block == "WM"`)

A localizer working-memory task run once at the start of the session before the main task.

**Design:** 24 trials — 4 shapes × 2 ISI conditions × 3 repetitions.

**Trial structure:** fixation → sample stimulus (0.5 s) → delay (ISI) → test display (4 slots) → response (deadline 2.0 s)

**WM-specific columns:**

| Column | Type | Description |
|---|---|---|
| `sample_stim` | string | Shape shown during sample phase (`circle`, `rectangle`, `star`, `triangle`) |
| `isi_condition` | float | Delay duration: `1.0` or `2.5` seconds |
| `t_sample` | float | Absolute time of sample stimulus onset |
| `t_delay` | float | Absolute time of delay (ISI) onset |

Columns not used in WM: `rule_type`, `A_stim`, `B_stim`, `A_prime`, `B_prime`, `rule_sequence`, `test_sequence`, `correct_next_stim`, `isi1`–`isi5`, `iti`, `t_rule1`–`t_rule3`, `t_test1`, `t_test2` (all empty).

---

### Main Task Blocks (`block == "main_runN"`)

Relational reasoning task run in 4 runs of 30 trials each (120 total).

**Trial structure:** fixation → rule sequence (3 stimuli, each ~1 s with jittered ISIs) → test sequence (2 stimuli) → response display (4 slots) → response → ITI

**Rule types:**
- `ABA`: the pattern follows an ABA structure (e.g., circle → triangle → circle); the correct next item follows the same rule applied to new exemplars
- `ABB`: the pattern follows an ABB structure (e.g., circle → triangle → triangle)

**Main-task-specific columns:**

| Column | Type | Description |
|---|---|---|
| `rule_type` | string | `"ABA"` or `"ABB"` |
| `A_stim` | string | Shape assigned to role A |
| `B_stim` | string | Shape assigned to role B |
| `A_prime` | string | New shape assigned to role A in test |
| `B_prime` | string | New shape assigned to role B in test |
| `rule_sequence` | string | Comma-separated 3-item sequence shown during rule phase (e.g., `"circle,triangle,circle"`) |
| `test_sequence` | string | Comma-separated 2-item sequence shown during test phase (e.g., `"triangle,circle"`) |
| `correct_next_stim` | string | The correct third item to complete the test sequence |
| `isi1`–`isi5` | float | Jittered ISI durations (s) between stimuli; drawn from truncated Normal(1.0, 0.2, min=0.0) |
| `iti` | float | Inter-trial interval (s); drawn from truncated Normal(3.0, 1.0, min=0.0) |
| `t_rule1` | float | Absolute time of first rule stimulus onset |
| `t_rule2` | float | Absolute time of second rule stimulus onset |
| `t_rule3` | float | Absolute time of third rule stimulus onset |
| `t_test1` | float | Absolute time of first test stimulus onset |
| `t_test2` | float | Absolute time of second test stimulus onset |

Columns not used in main task: `sample_stim`, `isi_condition`, `t_sample`, `t_delay` (all empty).

---

## Log File

PsychoPy plain-text log. Each line:

```
<elapsed_seconds>    <LEVEL>    <message>
```

Levels: `INFO`, `WARNING`, `EXP`, `DATA`

Key `DATA` entries:
- `Keypress: equal` — scanner TTL pulse (TR trigger)
- `Keypress: <1–4>` — participant response
- `WM trial N: sample=<shape>, isi=<duration>` — WM trial start
- `WM trial N response: key=<k>, rt=<rt>, correct=<0|1>` — WM trial response

---

## Stimuli

Four shapes, stored as PNG images in `assets/`:
- `circle.png`
- `rectangle.png`
- `star.png`
- `triangle.png`

---

## Sessions in `data/`

| Folder | UUID | Notes |
|---|---|---|
| `2026-03-02_15-10-18` | `cc2bdacf-...` | Full session |
| `2026-03-02_16-25-49` | `7ea7ce2b-...` | Full session |
| `2026-03-04_18-13-59` | `e3f4c2c2-...` | Full session |
