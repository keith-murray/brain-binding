# OPM-MEG Analysis Plan: Rule Phase Dynamics in the PerceptXBind Paradigm

## Background and Scientific Question

The perceptXbind task presents participants with two sequentially structured phases. In the **rule phase**, three stimuli (S₁, S₂, S₃) are presented in either an ABA or ABB pattern. In the **test phase**, a second three-stimulus sequence is presented, and the participant indicates whether it instantiates the same abstract rule. Test sequences can deviate from the rule phase in two ways: by a different in-distribution rule (ABB following ABA, or vice versa), or by an out-of-distribution structure (ABC).

This document focuses on the **rule phase** analyses. The central question is how the brain infers the abstract rule from the three-stimulus sequence — specifically, whether rule inference proceeds via a routing-and-comparison mechanism in which S₃ is compared against both maintained working memory (WM) traces, with selective persistence of the matching trace serving as the substrate of the inferred rule.

## Conceptual Framework

During the rule phase, the brain is hypothesized to:
1. Encode S₁ and S₂ into working memory, producing neural states **m**₁ and **m**₂ occupying (potentially separable) subspaces of the population activity.
2. Upon S₃ presentation, compute a comparison between S₃ and each WM trace, producing match signals $\text{match}_i = \langle f(\mathbf{s}_3), \mathbf{m}_i \rangle$ for $i \in \{1, 2\}$, where $f(\cdot)$ is the transform bringing S₃ into the comparison subspace.
3. Resolve the comparison into a function vector — the abstract rule ABA vs. ABB — corresponding to which positional slot S₃ maps onto.

The naive routing hypothesis predicts that both WM subspaces are transiently engaged during the comparison, followed by selective persistence of the matching trace. An alternative is that S₃ is compared directly against only the relevant WM trace, with no transient engagement of the non-matching subspace.

A secondary distinction of interest is the **format of WM maintenance**: whether S₁ and S₂ are stored compositionally in independent subspaces, or conjunctively as a joint S₁⊗S₂ code.

## Data and Modality

- **Recording modality**: OPM-MEG, providing millisecond temporal resolution and elevated SNR for cortical sources relative to SQUID-MEG due to on-scalp sensor placement.
- **Preprocessing**: Standard MNE-Python pipeline producing ICA-cleaned, epoched data. Epochs will be aligned to stimulus onsets (S₁, S₂, S₃) within each trial.

## Analysis 1: Rule Decodability at S₃

**Goal**: Establish that the abstract rule (ABA vs. ABB) is linearly decodable from sensor-space patterns following S₃ onset.

**Method**: At each time point $t$ relative to S₃ onset, train a binary linear classifier (logistic regression) on the sensor pattern $\mathbf{x}(t) \in \mathbb{R}^C$ to discriminate ABA vs. ABB trials. Use k-fold cross-validation within subject. Produce a decoding accuracy time course $d(t)$.

**Predictions**:
- Above-chance decoding emerging approximately 150–300 ms post-S₃ onset, consistent with timescales reported in recent OPM-MEG decoding literature (Xu et al., 2024).
- Sustained decoding into the inter-phase interval if the rule code is maintained for the upcoming test phase.

**Temporal generalization matrix (TGM)**: Extend the above by training a classifier at each $t$ and testing at each $t'$, yielding a 2D matrix $d(t, t')$.
- A **diagonal** TGM indicates a sequence of transient, non-recurring codes — consistent with active computation.
- A **square block** TGM indicates a stable, sustained code — consistent with an attractor-like rule representation.
- The prediction is a transition from diagonal structure during the comparison window (~100–300 ms) to a sustained block during the maintenance interval.

## Analysis 2: Working Memory Maintenance Verification

**Goal**: Before testing the routing hypothesis, establish that S₁ and S₂ item identities are maintained in decodable form through the S₃ window. This is a prerequisite analysis and also distinguishes compositional vs. conjunctive WM formats.

**Method**:
- Train a shape-identity decoder $D_1$ on the S₁ encoding epoch.
- Test $D_1$ via temporal generalization onto the S₂ epoch and the S₂–S₃ maintenance interval.
- Repeat for $D_2$ trained on S₂.

**Predictions**:
- **Compositional format**: $D_1$ trained during S₁ encoding generalizes to later time points during S₂ presentation and the maintenance interval — the S₁ identity persists even while S₂ is being encoded.
- **Conjunctive format**: $D_1$ fails to generalize beyond the S₁ epoch; the code format shifts when S₂ arrives, and a joint-pair decoder would be required instead.

## Analysis 3: Routing and Selective Persistence (Primary Hypothesis Test)

**Goal**: Test whether S₃ comparison engages both WM subspaces transiently, with selective persistence of the matching trace.

### Confound to address first

An important structural feature of the task: in ABA trials, S₃ is physically identical to S₁; in ABB trials, S₃ is physically identical to S₂. Consequently, cross-decoding from the matching position is trivially supported by sensory-level stimulus identity. The routing hypothesis lives specifically in a narrower signal: **transient reactivation of the non-matching WM trace during the comparison window**.

### Method: cross-decoding approach

Using decoders $D_1$ (trained on S₁ encoding) and $D_2$ (trained on S₂ encoding):
- For **ABA trials**: Apply $D_2$ to the S₃ epoch. Since S₃ ≠ S₂ at the sensory level, sensory-driven decoding should be at chance. Any above-chance transient would reflect reactivation of **m**₂ during comparison.
- For **ABB trials**: Apply $D_1$ to the S₃ epoch. Symmetric logic.

### Method: template similarity approach (complementary)

Compute time-resolved similarity between the S₃ sensor pattern and templates from the encoding epochs:

$$\rho_{S_1}(t) = \text{corr}\big(\mathbf{x}_{S_3}(t), \, \bar{\mathbf{x}}_{S_1}^{\text{template}}\big)$$

$$\rho_{S_2}(t) = \text{corr}\big(\mathbf{x}_{S_3}(t), \, \bar{\mathbf{x}}_{S_2}^{\text{template}}\big)$$

This provides a continuous similarity measure at each time point, more sensitive than binary classification for detecting weak reactivation.

### Predictions under competing hypotheses

**H1: Routing to both + selective persistence**
- Matching decoder on S₃: above chance from ~100 ms, sustained.
- Non-matching decoder on S₃: brief transient above chance (~100–250 ms), returns to chance.
- TGM for the rule decoder: diagonal during the comparison window, transitioning to a sustained block.

**H2: Direct match only (no retrieval of non-matching item)**
- Matching decoder: above chance, sustained.
- Non-matching decoder: never above chance.
- The rule decoder still emerges but via a different computational mechanism.

**H3: Conjunctive WM code**
- $D_1$ and $D_2$ fail to generalize from encoding to maintenance (falsified in Analysis 2).
- A joint-pair decoder would be required; the current routing prediction does not apply.

The H1–H2 distinction is the critical test of the routing hypothesis. Null results on H1 cannot distinguish H2 from insufficient statistical power, so trial counts per condition are a limiting factor (rough target: ≥80–100 trials per rule type per participant).

## Analysis 4: Time-Resolved RSA with Multiple Model RDMs

**Goal**: Characterize the geometry of the evolving representation by decomposing sensor-pattern similarity into multiple hypothesized structures.

**Method**: At each time point $t$, construct a neural RDM across conditions from pairwise distances of sensor patterns. Regress this neural RDM against a set of model RDMs:
- **Stimulus identity**: which shape is presented at each position.
- **Position-match-to-S₁**: indicator of whether the current stimulus matches S₁.
- **Position-match-to-S₂**: indicator of whether the current stimulus matches S₂.
- **Rule**: ABA vs. ABB.

The temporal profiles of the resulting regression coefficients $\beta_k(t)$ track when each representational dimension is expressed.

**Predictions**:
- Stimulus identity dominant at early latencies following each stimulus onset.
- Match-to-S₁ and match-to-S₂ significant at intermediate latencies during the S₃ window.
- Rule representation emerging at later latencies and persisting into maintenance.

This analysis is the temporal analogue of the RDM-space regression already used in the fMRI pipeline, and provides a natural bridge for cross-modal fusion analyses downstream.

## Statistical Inference

- Cluster-based permutation testing across time (and across the TGM where applicable) following Maris & Oostenveld (2007), cluster-defining threshold $p < 0.05$, cluster-corrected significance $p < 0.05$.
- Group-level inference on subject-wise time courses using a second-level permutation test against chance.
- For cross-decoding transients, a priori time windows of interest (~100–300 ms post-S₃) will reduce multiple comparison burden.

## Relationship to the fMRI Pipeline

The sensor-space MVPA analyses here are temporally complementary to the searchlight RSA already implemented in the fMRI pipeline. The shared model RDM construction (stimulus, role, rule, match-position) allows subsequent cross-modal RSA fusion (Cichy et al., 2014), in which time-resolved MEG RDMs are correlated against searchlight fMRI RDMs to produce spatiotemporal maps of shared representational structure.

## Key References

- Cichy, R. M., Pantazis, D., & Oliva, A. (2014). Resolving human object recognition in space and time. *Nature Neuroscience*.
- King, J.-R., & Dehaene, S. (2014). Characterizing the dynamics of mental representations: the temporal generalization method. *Trends in Cognitive Sciences*.
- Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG-data. *Journal of Neuroscience Methods*.
- Seymour, R. A., et al. (2022). Interference suppression techniques for OPM-based MEG. *NeuroImage*.
- van Driel, J., et al. (2021). High-pass filtering artifacts in multivariate classification. *Journal of Neuroscience Methods*.
- Xu, Y., et al. (2024). Decoding the temporal structures and interactions of multiple face dimensions using OPM-MEG. *Journal of Neuroscience*.