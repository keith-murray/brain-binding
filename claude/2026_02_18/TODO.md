# fMRI Class Project: Relational Reasoning Task (ABA vs ABB)

## Overview

This class project will collect pilot fMRI data from **2 participants**. Each participant completes:

1. **Working-memory (WM) control task** (~3 min)
2. **Rest** (2 min)
3. **Main relational reasoning task** (~25 min; **120 trials**)

The experiment uses a fixed set of **4 visual stimuli** (denoted S1–S4). The main task tests relational pattern completion for **ABA** vs **ABB** sequences.

---

## Stimulus Set and Response Interface

### Stimuli
- A set of **4 distinct stimuli**: **S1, S2, S3, S4**
- All stimuli are assumed to be visually discriminable and matched as closely as possible for low-level properties (size, luminance, contrast).

### Response mapping (both tasks)
- Participant responds with **4-button input** selecting among **S1–S4**.
- On each trial’s response screen, the four stimuli are displayed in **four spatial slots**.
- The **mapping from stimulus identity → slot position** is **randomized each trial** (independently), to reduce fixed motor confounds.
- Participant presses the corresponding button for the chosen slot.
- Response deadline: **2.0 s cutoff** (both tasks unless otherwise stated).

---

## Session Timeline (per participant)

|           Block | Duration | Notes                   |
| --------------: | :------- | :---------------------- |
| WM control task | ~3 min   | 24 trials               |
|            Rest | 2 min    | fixation / passive rest |
|       Main task | ~25 min  | 120 trials              |

---

## Block 1: WM Control Task

### Goal
Provide a within-subject control for:
- maintaining stimulus identity across a delay,
- and comparing **short vs longer effective maintenance** durations.

The WM task will include two delay variants:
- **Short delay:** ISI = **1.0 s**
- **Long delay:** ISI = **2.5 s** (intended to mimic the additional time that would be occupied by an extra stimulus presentation in the main task)

### Trial Structure
Each trial consists of:

1. **Fixation / trial start cue**
   - duration: **0.5s**
   - Display: white fixation dot
   - Duration: fixed or minimal (implementation-dependent; see Timing Implementation Notes)

2. **Sample stimulus**
   - Duration: **0.5 s**

3. **Delay (ISI)**
   - Either **1.0 s** or **2.5 s**
   - **Randomly permuted** across trials (balanced counts; see Counterbalancing)

4. **Response screen (4AFC)**
   - Four stimuli displayed in four slots (randomized positions)
   - Response cutoff: **2.0 s**
   - Outcome: accuracy + RT recorded

### Trial Counts and Counterbalancing
Total trials: **24**

Design target:
- For each of the 4 stimuli, repeat **3 times** at each ISI condition:
  - Stimulus identity: 4
  - ISI condition: 2 (1.0 s vs 2.5 s)
  - Repetitions: 3
  - Total: 4 × 2 × 3 = **24 trials**

Randomization constraints:
- ISI conditions are **randomly permuted** trial-to-trial.
- Stimulus identities are **balanced** across the task.
- (Optional) Avoid immediate repetitions of the same stimulus identity if desired.

### Outputs
- Trialwise: sample stimulus ID, ISI condition, response (chosen stimulus), correctness, RT, response mapping (stimulus→slot).

---

## Block 2: Rest

- Duration: **2 minutes**
- Display: fixation dot
- Instruction: rest, minimize motion, maintain fixation.

---

## Block 3: Main Relational Reasoning Task (ABA vs ABB)

### Goal
On each trial, participants observe a **rule phase** (3 items) following either **ABA** or **ABB** using the same 4 stimuli, followed by a **test phase** containing **AB** (2 items). The participant reports the **next stimulus** in the sequence (the 3rd item consistent with the inferred rule) via 4AFC.

### Trial Structure

**A. Trial start**
- Display: white fixation dot
- Onset indicates start of trial (see Timing Implementation Notes)

**B. Rule phase (3 stimuli)**
- Stimulus 1: 0.5 s
- ISI: 1.0 s + Gaussian jitter (mean 0, sd 0.2 s)
- Stimulus 2: 0.5 s
- ISI: 1.0 s + Gaussian jitter (mean 0, sd 0.2 s)
- Stimulus 3: 0.5 s
- ISI (before test phase): 1.0 s + Gaussian jitter (mean 0, sd 0.2 s) 

Rule type:
- **ABA**: item1 = A, item2 = B, item3 = A
- **ABB**: item1 = A, item2 = B, item3 = B

Constraint:
- **A ≠ B** (A and B must be different stimuli)

**C. Test phase (2 stimuli; AB)**
- Test item1 (A): 0.5 s
- ISI: 1.0 s + Gaussian jitter (mean 0, sd 0.2 s)
- Test item2 (B): 0.5 s

**D. Response phase (4AFC)**
- Prompt: “Select the next stimulus”
- Four stimuli displayed in four slots (randomized positions)
- Response cutoff: **2.0 s**

Correct response:
- If rule was **ABA**, next item after test AB is **A**
- If rule was **ABB**, next item after test AB is **B**

**E. Inter-trial interval (ITI)**
- Base ITI: **2.5 s**
- Jitter: **+ Gaussian(0, 1.0 s)**
- Display: fixation dot

> Note: If Gaussian jitter yields negative values, apply truncation at 0 (or a small minimum), e.g. ITI = max(0, 3.0 + N(0,1)).

### Timing Summary (Main Task)

| Component       |            Duration |
| :-------------- | ------------------: |
| Each stimulus   |               0.5 s |
| Each ISI        | 1.0 s + N(0, 0.2 s) |
| Response cutoff |               2.0 s |
| ITI             | 3.0 s + N(0, 1.0 s) |

Per trial, excluding jitter, the nominal timeline is:
- 5 stimuli × 0.5 s = 2.5 s
- 3 ISIs × 1.0 s = 3.0 s (between the 5 stimuli: after rule1, rule2, and test1)
- Response = 2.0 s
- ITI = 3.0 s
- Nominal total ≈ **10.5 s/trial** (plus jitter)

With **120 trials**, total nominal time ≈ **21 minutes** plus jitter and any fixed fixation-onsets, aligning with the ~25 min target.

---

## Main Task Trial Generation and Counterbalancing

### Core combinatorics (4 stimuli)
For a given trial:
- Choose A from 4 stimuli
- Choose B from remaining 3 stimuli
- This yields **12 ordered (A,B) pairs**

For each (A,B) pair, two rules exist:
- ABA and ABB

Total unique instantiations at the (A,B,rule) level:
- 12 pairs × 2 rules = **24 unique configurations**

### 120-trial plan (no generalization set)
Given 120 trials, a simple balanced approach is:

- Repeat each of the 24 unique configurations **5 times**:
  - 24 × 5 = 120

This implies:
- Each ordered (A,B) pair appears 10 times total (5 under ABA + 5 under ABB)
- ABA and ABB appear equally often (60 each)

Randomization constraints (recommended):
- Shuffle trials within runs/blocks.
- Avoid long streaks of the same rule (e.g., no more than 3 in a row).
- Avoid immediate repetition of identical (A,B,rule) configurations when possible.

Recorded factors per trial:
- Rule type (ABA vs ABB)
- A identity (S1–S4)
- B identity (S1–S4, B≠A)
- Response mapping (stimulus→slot)
- Correct response identity (A or B)
- Choice, correctness, RT
- Realized jitters for each ISI and ITI

---

## Data Quality Notes and Known Limitations (Class Project Constraints)

1. **Sample size**
   - N=2 is strictly pilot/demonstration; inference about population effects will be limited.

2. **Relational vs repetition confound**
   - ABB includes an immediate repetition in the rule phase (positions 2–3), whereas ABA does not.
   - Any “rule decoding” may partly reflect repetition detection/suppression rather than abstract relation encoding.

3. **Stimulus-set size and learning**
   - With only 4 stimuli and repeated configurations (24 configs repeated 5×), participants may learn pair→response mappings.
   - This reduces “pure generalization” claims; acceptable for a class demo but should be stated explicitly.

4. **Motor confounds mitigated by randomized response mapping**
   - Randomizing stimulus positions on the response screen reduces fixed mapping between identity and motor output.
   - Still, response-related activity will be present; model response screen separately in analyses if doing GLM/MVPA.

5. **Gaussian jitter**
   - Gaussian jitter can produce negative durations.
   - Implementation should apply truncation (e.g., clip at 0) or resample until positive.

---

## Implementation Notes (Practical)

- Use a consistent definition of trial onset (e.g., fixation onset).
- Log actual flip times for all stimuli and response screens.
- Ensure the ISI/ITI sampling is done in seconds and applied consistently.
- Store random seeds per participant for reproducibility.
- Consider splitting the 120 trials into multiple runs (e.g., 4×30 trials) if scanner workflow requires it.

---

## Output Logs (Recommended Columns)

### WM task
- participant_id
- trial_index
- sample_stim (S1–S4)
- isi_condition (1.0 or 2.5)
- response_slot (1–4)
- response_stim (S1–S4)
- correct (0/1)
- rt
- slot_mapping (e.g., {slot1:S3, slot2:S1, slot3:S4, slot4:S2})
- onset times (fixation, sample, response)

### Main task
- participant_id
- trial_index
- rule_type (ABA/ABB)
- A_stim, B_stim
- rule_sequence (e.g., S2,S4,S2)
- test_sequence (e.g., S2,S4)
- correct_next_stim (A_stim or B_stim)
- isi_jitters (list of realized ISIs)
- iti_jitter (realized ITI)
- response_slot, response_stim, correct, rt
- slot_mapping
- onset times (each stimulus, response, ITI/fixation)

---