# RSA and Encoding Models Cookbook — perceptXbind MEG

## Purpose and scope

This cookbook specifies how to add two complementary analysis methods to the project: **representational similarity analysis (RSA)** and **encoding models**. Both operate in the information-presence framework — they ask whether and how task-relevant variables are encoded in MEG population activity, complementing the decoding toolkit's classification-based approach.

The student should implement these as new modules in the existing toolkit, sharing the data-loading and CV machinery that's already in place. Implementation order: RSA first (lighter and closer to existing infrastructure), then encoding models.

This document does **not** cover the data preprocessing pipeline — that's already done. It assumes the student has access to ICA-cleaned, epoched data from `tim/ported_results/sub-001_ses-01_task-binding_epo.fif` and the trial-aligned behavioral CSV.

---

## 1. Conceptual frame

Both RSA and encoding models extend the basic question of decoding ("can class labels be predicted from neural activity?") to richer questions about representational structure.

**RSA** asks: *given a set of conditions, how similar are their neural patterns to each other, and does that similarity structure match a hypothesized model?* You compute pairwise distances between condition-mean neural patterns to get a neural RDM, then compare it to model RDMs encoding hypothesized representational structures.

**Encoding models** ask: *can a linear (or other) function of features in some hypothesized feature space predict neural activity at each sensor/voxel?* The model is fit on training trials and evaluated on held-out trials; success implies that feature space captures something the brain encodes.

The two are deeply related — under linear encoding models with squared-error loss, the encoding R² in a region is mathematically equivalent to a particular RSA correlation up to scaling. In practice, RSA is faster and gives a single summary number per ROI/timepoint; encoding models provide richer, voxel/sensor-resolved predictions and naturally extend to continuous predictors.

For our project, the analytical question structure is:

| Question | Best method |
|---|---|
| Is rule type encoded at this time? | Decoding (already implemented) |
| Is rule type encoded *similarly* to how rule-position is encoded? | RSA |
| Does a model where rule = "pointer to position" predict neural activity? | Encoding model |
| How does the geometry of representations evolve over time? | Time-resolved RSA |

---

## 2. Condition definition (shared by both methods)

This is the most important upstream step and is shared by RSA and encoding models. Get this wrong and everything downstream is meaningless.

A **condition** is a discrete combination of task variables that occurs across multiple trials. The student should construct a condition coding scheme that aligns with the hypotheses being tested.

For perceptXbind, the relevant condition variables are:

| Variable | Levels | Source |
|---|---|---|
| `phase` | rule1, rule2, rule3, test1, test2, test3 | within-trial position |
| `rule_type` | ABA, ABB | `df['rule_type']` |
| `test_type` | ABA, ABB, other | `df['test_type']` |
| `stim_identity` | circle, rectangle, star, triangle | derived from sequence position |
| `role_in_rule` | A, B (for rule phase) | computed from rule_type and position |
| `role_in_test` | X, Y, X′ (for test phase) | computed from test_type and position |
| `match` | match, mismatch | `df['match']` |

For RSA, conditions are typically defined by *combinations* of these variables. Example schemes:

- **Stimulus-by-role** (16 conditions for rule phase): {circle, rectangle, star, triangle} × {A-role, B-role}. This is what tests whether shape and role are encoded compositionally vs. conjunctively.
- **Phase-by-rule** (12 conditions): {rule1, rule2, rule3, test1, test2, test3} × {ABA, ABB}. Tests rule-dependent dynamics.
- **Stim-by-phase-by-role** (could be 4 × 6 × 2 = 48 conditions): the most fine-grained, but with ~2–3 trials per condition gets noisy.

The student should **start with the stimulus-by-role scheme** for the rule phase, since it directly maps to our scientific questions. Then expand to other schemes as needed.

### Condition labels per epoch

Implement a function that takes the behavioral CSV and the within-trial epoch position and returns a condition label per epoch:

```python
def assign_conditions(
    df: pd.DataFrame,                 # behavioral, 120 rows
    scheme: str = "stim_by_role",     # which condition scheme
) -> Dict[str, np.ndarray]:
    """
    Returns a dict with keys:
      'condition_id' : (1080,) integer condition labels per epoch
      'condition_names' : list of human-readable names indexed by id
      'epoch_phase' : (1080,) string phase label
      'condition_metadata' : DataFrame with one row per unique condition
    """
```

Conditions should be assigned for *all 1080 epochs*; the caller filters to the relevant epoch subset (e.g., only rule3 epochs) before passing to RSA or encoding routines.

The `condition_metadata` DataFrame is critical — it's what model RDMs and encoding feature matrices are built from. Each row should describe one unique condition with all its variable values:

```
cond_id  shape       role  rule_type  phase
0        circle      A     ABA        rule1
1        circle      A     ABA        rule3
...
```

### Trial counts per condition

After assigning conditions, the student should print per-condition trial counts as a sanity check. With 120 trials and 16 conditions, expect ~7–8 trials per condition on average, but balance varies by scheme. Conditions with fewer than ~5 trials should be flagged or merged before RSA.

---

## 3. Representational Similarity Analysis (RSA)

### 3.1 Pipeline overview

The RSA pipeline has four stages:

1. **Compute neural patterns** — one vector per condition, per timepoint (or aggregated time window). This is just the condition-averaged sensor pattern.
2. **Compute neural RDM** — pairwise distances between all condition patterns.
3. **Construct model RDMs** — encode hypothesized representational structures as pairwise dissimilarity matrices.
4. **Compare** — correlate or regress neural RDM against model RDMs.

Each stage has design decisions worth getting right.

### 3.2 Neural pattern computation

For each condition $c$ and timepoint $t$, compute:

$$\bar{\mathbf{x}}_c(t) = \frac{1}{|S_c|} \sum_{i \in S_c} \mathbf{x}_i(t)$$

where $S_c$ is the set of trials assigned to condition $c$ and $\mathbf{x}_i(t) \in \mathbb{R}^C$ is the sensor pattern.

**Key point**: condition-averaging respects CV folds the same way pseudo-trial averaging does. For cross-validated RSA (covered below), patterns are computed within fold splits.

For the rule phase analyses, the student can choose either:

- **Time-resolved**: one neural RDM per timepoint, giving an RDM time course
- **Time-averaged**: average activity in a window (e.g., 100–300ms post-S₃), one RDM per analysis

Start with time-resolved. The time-averaged version is a special case if you reduce the time axis upstream.

### 3.3 Neural RDM construction

Pairwise distance between condition patterns. Recommended distance metrics, in order of preference:

**1. Crossnobis (cross-validated Mahalanobis distance) — preferred.**

This is the standard for RSA in modern cognitive neuroscience. For two conditions $c_1$ and $c_2$ across two CV folds A and B:

$$d_{c_1, c_2}^{\text{xnobis}} = (\bar{\mathbf{x}}_{c_1}^A - \bar{\mathbf{x}}_{c_2}^A)^\top \Sigma^{-1} (\bar{\mathbf{x}}_{c_1}^B - \bar{\mathbf{x}}_{c_2}^B)$$

where $\Sigma$ is the residual covariance matrix from the fold-A patterns. The cross-fold structure makes the estimator unbiased — under the null of identical conditions, expected distance is exactly zero (regular Mahalanobis distance is positively biased). Crossnobis distances can therefore be negative, which is a feature, not a bug.

For implementation, use the `rsatoolbox` Python package (`pyrsa`) which has a tested crossnobis estimator. Reinventing this is a common source of subtle bugs.

**2. Pearson correlation distance.**

$$d_{c_1, c_2}^{\text{corr}} = 1 - \rho(\bar{\mathbf{x}}_{c_1}, \bar{\mathbf{x}}_{c_2})$$

Faster, no covariance estimation needed. Insensitive to gain differences, which is sometimes desirable. Good as a quick alternative if `rsatoolbox` setup is delayed.

**3. Euclidean / Mahalanobis (non-crossvalidated).** Avoid. Positively biased under the null and harder to interpret.

The output is a $K \times K$ symmetric matrix per timepoint, where $K$ is the number of conditions. Store as `(n_times, K, K)` array.

### 3.4 Model RDM construction

For each hypothesis about representational structure, construct a $K \times K$ model RDM where entry $(i, j)$ is the predicted dissimilarity between conditions $i$ and $j$ under that hypothesis.

For the rule phase analyses, build at least these models:

| Model RDM | Definition | What it tests |
|---|---|---|
| **Stimulus identity** | 0 if $\text{shape}(i) = \text{shape}(j)$, else 1 | Sensory shape coding |
| **Abstract role** | 0 if $\text{role}(i) = \text{role}(j)$, else 1 | Role-bound coding (A vs B abstractly) |
| **Concrete role** | distinguishes (A=circle) from (A=triangle) etc. | Conjunctive shape×role coding |
| **Rule type** | 0 if $\text{rule}(i) = \text{rule}(j)$, else 1 | Rule representation |
| **Position** | 0 if same within-trial position, else 1 | Position/timing coding |
| **Conjunctive binding** | 0 only if shape AND role match | Tests for binding-specific code |

Implement as a function:

```python
def build_model_rdm(
    condition_metadata: pd.DataFrame,    # one row per condition
    model_name: str,
) -> np.ndarray:                         # (K, K) symmetric, zero diagonal
```

**Critical sanity check**: model RDMs should not be perfectly collinear with each other, or you can't separate their contributions in regression. Compute the variance inflation factor across your model RDMs:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Vectorize the upper triangle of each model RDM
X_models = np.stack([rdm[np.triu_indices(K, k=1)] for rdm in model_rdms], axis=1)
vifs = [variance_inflation_factor(X_models, i) for i in range(X_models.shape[1])]
```

VIFs above ~5 indicate problematic collinearity. In our fMRI pipeline this came up — the position and abstract-role regressors were perfectly collinear, so position was dropped. The student should expect to do a similar check and possibly drop one of the candidate models.

### 3.5 Comparing neural and model RDMs

Two approaches:

**Simple Spearman RSA.** For each timepoint, correlate the upper triangle of the neural RDM with the upper triangle of each model RDM:

$$\rho_m(t) = \text{Spearman}\big(\text{vec}(\text{NeuralRDM}(t)), \text{vec}(\text{ModelRDM}_m)\big)$$

Spearman is preferred over Pearson because it's non-parametric and the neural RDM's distance values are not well-behaved on a linear scale.

Use this when models are clearly distinct and you want an interpretable per-model time course. Output: $\rho_m(t)$ for each model $m$.

**Partial RDM-space regression.** For each timepoint, fit a linear regression over RDM-space pairs:

$$y_{ij}(t) = \sum_m \beta_m(t) \cdot x_{ij}^{(m)} + \epsilon_{ij}(t)$$

where $y_{ij}(t)$ are the neural pairwise distances and $x_{ij}^{(m)}$ are model $m$'s pairwise distances. The $\beta_m(t)$ values quantify each model's *unique* contribution after controlling for the others.

Use this when models are correlated and you want to dissociate their contributions. Output: $\beta_m(t)$ for each model.

A practical note from our fMRI pipeline: rankdata should be applied to the neural distances $y_{ij}$ but **not** to binary model RDM vectors $x_{ij}^{(m)}$. Ranking binary vectors destroys their interpretability. Raw $y$ values are also acceptable and more interpretable; only rank if the residual distribution is badly non-Gaussian.

### 3.6 Time-resolved RSA implementation

Putting it together, the time-resolved RSA pipeline produces a $(\text{n\_models}, \text{n\_times})$ matrix of fits. Suggested function:

```python
def time_resolved_rsa(
    X: np.ndarray,                       # (n_epochs, n_channels, n_times)
    condition_ids: np.ndarray,           # (n_epochs,) condition label per epoch
    model_rdms: Dict[str, np.ndarray],   # {model_name: (K, K) model RDM}
    method: str = "regression",          # "spearman" | "regression"
    distance: str = "crossnobis",        # "crossnobis" | "correlation"
    cv_splitter: Optional[CVSplitter] = None,  # required for crossnobis
    rng: Optional[np.random.Generator] = None,
) -> RSAResult:
```

`RSAResult` should be a dataclass paralleling `CVResult`:

```python
@dataclass
class RSAResult:
    fits: np.ndarray                     # (n_models, n_times) Spearman ρ or regression β
    model_names: List[str]
    times: np.ndarray
    neural_rdms: Optional[np.ndarray]    # (n_times, K, K) for inspection
    method: str
    distance: str
```

### 3.7 Statistical inference for RSA

Use the same cluster-based permutation testing as for decoding:

```python
from mne.stats import permutation_cluster_1samp_test

# fits has shape (n_subjects, n_times) for one model
T_obs, clusters, p_vals, H0 = permutation_cluster_1samp_test(
    fits, threshold=None, n_permutations=10000, tail=1
)
```

For single-subject preliminary work (which is what we're doing now), instead permute the condition labels within the original data and recompute the RSA fits, building an empirical null distribution per timepoint.

### 3.8 Encoding-space RSA (advanced)

Once basic time-resolved RSA is working, the student can add a more powerful variant: instead of computing RDMs from condition means, compute them from encoding model predictions, separating out which feature dimensions co-activate. This was implemented in our fMRI pipeline as the "condition-space encoding model." Details deferred to v2.

---

## 4. Encoding models

### 4.1 Pipeline overview

An encoding model predicts neural activity from a feature representation:

$$\mathbf{x}_i(t) = \mathbf{W}(t) \boldsymbol{\phi}(\mathbf{s}_i) + \boldsymbol{\eta}_i(t)$$

where $\boldsymbol{\phi}(\mathbf{s}_i) \in \mathbb{R}^F$ is a feature vector for trial $i$ and $\mathbf{W}(t) \in \mathbb{R}^{C \times F}$ is the weight matrix at time $t$. Fit $\mathbf{W}$ on training trials, predict held-out trials, evaluate prediction quality.

Encoding models complement RSA by:
- Naturally handling continuous predictors (RT, coherence) that don't fit a discrete-condition framework
- Producing sensor-resolved predictions (a topographic map of model fit)
- Allowing nested model comparison (does adding feature X improve prediction beyond features Y, Z?)

### 4.2 Feature space construction

The student should implement at least these feature spaces for the rule phase:

**Stimulus identity (one-hot)**: 4-dimensional, one-hot encoding of the current stimulus shape.

**Stimulus identity × position**: 12-dimensional, one-hot encoding of the (shape, position) conjunction. Tests for position-binding.

**Stimulus identity × role**: 8-dimensional, one-hot encoding of (shape, role). Tests for role-binding.

**Rule type**: 1-dimensional binary indicator (or 2-dimensional one-hot).

**Combined model**: stack the above for nested comparison.

For each, the feature representation is computed per trial from the behavioral metadata. Implement as:

```python
def build_feature_space(
    df: pd.DataFrame,                    # 120 rows, behavioral
    epoch_phase: np.ndarray,             # (1080,) within-trial phase per epoch
    feature_name: str,
) -> np.ndarray:                         # (n_epochs, n_features)
```

### 4.3 Fitting the encoding model

Fit a separate ridge regression per sensor and per timepoint:

$$\hat{\mathbf{w}}_{c,t} = (\Phi^\top \Phi + \lambda I)^{-1} \Phi^\top \mathbf{x}_{c,t}$$

where $\Phi \in \mathbb{R}^{N \times F}$ is the feature matrix (rows = trials, columns = features) and $\mathbf{x}_{c,t}$ is the activity at sensor $c$, time $t$, across trials.

Use `sklearn.linear_model.Ridge` (closed-form, fast) or `RidgeCV` (cross-validated $\lambda$). For our small sample size, $\lambda$ matters a lot — use `RidgeCV` with a log-spaced grid `np.logspace(-3, 3, 20)` and an inner CV fold structure.

**Critical**: $\lambda$ selection is part of fitting and must happen on the training set only — `RidgeCV` does this correctly.

### 4.4 Evaluation

Held-out prediction quality, evaluated per-sensor per-timepoint:

$$R^2_{c,t} = 1 - \frac{\sum_{i \in \text{test}}(x_{i,c,t} - \hat{x}_{i,c,t})^2}{\sum_{i \in \text{test}}(x_{i,c,t} - \bar{x}_{c,t})^2}$$

Pearson correlation between predicted and observed is also commonly reported. Both are positively biased in finite samples; the student should report both and note that R² in noise-dominated regions can be slightly negative (this is fine and expected).

Output shape: $(n_{\text{folds}}, n_{\text{channels}}, n_{\text{times}})$ for $R^2$.

### 4.5 Nested model comparison

The most analytically powerful use of encoding models is comparing nested feature spaces:

- Model A: stimulus identity only (4 features)
- Model B: stimulus identity + role (8 features)

The unique variance explained by adding role is $R^2_B - R^2_A$, computed per sensor per timepoint. Significantly positive values indicate that role information improves prediction beyond stimulus identity — direct evidence for role coding.

This is the encoding-model analog of partial RDM regression and addresses the same question.

### 4.6 Encoding model implementation

```python
def fit_encoding_model(
    X: np.ndarray,                       # (n_epochs, n_channels, n_times)
    Phi: np.ndarray,                     # (n_epochs, n_features) feature matrix
    cv: CVSplitter,
    alpha_grid: np.ndarray = np.logspace(-3, 3, 20),
    score: str = "r2",                   # "r2" | "pearson"
) -> EncodingResult:
```

```python
@dataclass
class EncodingResult:
    scores: np.ndarray                   # (n_folds, n_channels, n_times)
    weights: Optional[np.ndarray]        # (n_folds, n_channels, n_features, n_times)
    alphas: np.ndarray                   # (n_folds, n_channels, n_times) selected λ
    feature_names: List[str]
```

For the nested model comparison case, run two `fit_encoding_model` calls and subtract:

```python
result_A = fit_encoding_model(X, Phi_A, cv)
result_B = fit_encoding_model(X, Phi_B, cv)
unique_variance = result_B.scores - result_A.scores
```

### 4.7 Statistical inference for encoding models

Same cluster-based permutation framework as RSA and decoding. For nested model comparison, the unit of analysis is the difference $R^2_B - R^2_A$ at each (channel, timepoint), tested against zero across folds (within-subject) or across subjects (group-level).

---

## 5. Cross-modal RSA (fMRI fusion)

The richest analysis combines RSA from the MEG and fMRI pipelines. The recipe (Cichy et al. 2014, Nat Neurosci):

1. Compute MEG RDM at each timepoint $t$: $\text{MEG-RDM}(t)$
2. Compute fMRI RDM at each searchlight voxel $v$ from the existing fMRI pipeline: $\text{fMRI-RDM}(v)$
3. Correlate: $\rho(v, t) = \text{Spearman}(\text{MEG-RDM}(t), \text{fMRI-RDM}(v))$

Result: a 4D map of where (in cortex) and when (in time) representational geometries align across modalities. Hotspots identify regions whose fMRI-measured representations match the MEG signal's geometry at a particular time, providing spatiotemporal localization that neither modality alone can give.

This requires that **MEG and fMRI use the same condition definitions**. The condition-coding scheme described in §2 should be applied to both pipelines. Coordinate this with whoever runs the fMRI pipeline.

Implementation should be straightforward once the per-modality RDMs are computed:

```python
def cross_modal_rsa(
    meg_rdms: np.ndarray,                # (n_times, K, K)
    fmri_rdms: np.ndarray,               # (n_voxels, K, K)
    method: str = "spearman",
) -> np.ndarray:                         # (n_voxels, n_times) similarity map
```

For our project this is the obvious paper-quality figure: the time course of cortical alignment between modalities. Defer until single-modality analyses are working, but design the condition coding now to support it.

---

## 6. Integration with existing toolkit

Both RSA and encoding models should reuse:

- `CVSplitter` for cross-validation
- The trial-to-epoch alignment logic from the data spec
- The Feature classes for any time-frequency or covariance preprocessing (encoding models can take TF features as easily as raw amplitudes)

Place new modules at:

```
toolkit/
  features.py              (existing)
  decoders.py              (existing)
  cv.py                    (existing)
  pseudo_trials.py         (existing)
  conditions.py            (NEW — condition coding)
  rsa.py                   (NEW — RSA pipeline)
  encoding.py              (NEW — encoding models)
  cross_modal.py           (NEW — fMRI-MEG fusion)
```

Tests:

```
tests/
  test_conditions.py       (verify condition assignment matches behavioral CSV)
  test_rsa.py              (synthetic data with known RDM structure)
  test_encoding.py         (synthetic data with known feature-to-activity map)
```

---

## 7. Order of implementation

For the student:

1. **Condition coding** (`conditions.py`) — must come first, both methods depend on it. Verify by printing condition counts and a `condition_metadata` table.
2. **Model RDM construction + VIF check** — verify the model RDMs are sensible and not collinear before any neural data work.
3. **Time-resolved RSA with crossnobis distance and Spearman comparison** — the simplest working RSA pipeline.
4. **Time-resolved RSA with partial regression** — extension once Spearman is working.
5. **Encoding model with single feature space** — simplest version, ridge per sensor per timepoint.
6. **Encoding model with nested comparison** — extension.
7. **Cross-modal RSA** — only after both single-modality pipelines work and the fMRI side has matching condition coding.

Each step should produce a runnable demo notebook on sub-001.

---

## 8. Practical cautions

1. **Condition coding is the most important step.** Spend time getting it right; print and visually inspect the condition metadata table before running any analyses.
2. **Crossnobis requires a CV split structure.** Use the existing `CVSplitter` and follow its train-test convention. Distance is computed using one fold's patterns × another fold's patterns.
3. **Model RDMs should not be near-collinear.** Run the VIF check before regression. If two models are perfectly collinear, drop one.
4. **Encoding model regularization matters.** Always use `RidgeCV` with a reasonable grid; unregularized ridge will overfit hard in our $d > N$ regime.
5. **Both methods extend naturally to time-frequency features.** Once basic raw-amplitude versions work, swap in TF features by reusing `TimeFrequencyFeature`.
6. **Cross-validation for both methods follows the same trial-level rule as the decoding toolkit:** original trials must not appear in both train and test. The condition averaging and feature construction happen *inside* the CV loop, not before.
7. **For RSA, use `rsatoolbox` (Python `pyrsa`) where possible.** It implements crossnobis correctly and integrates with cluster permutation testing. Don't reinvent crossnobis from scratch — it's easy to get the cross-fold structure wrong.

---

## 9. References for the student

- Kriegeskorte, Mur, & Bandettini (2008), *Frontiers in Systems Neuroscience* — foundational RSA paper
- Diedrichsen & Kriegeskorte (2017), *PLoS Comp Bio* — modern RSA and crossnobis
- Cichy, Pantazis, & Oliva (2014), *Nat Neurosci* — MEG-fMRI fusion via RSA
- Naselaris et al. (2011), *NeuroImage* — encoding vs decoding models, foundational reference
- Nili et al. (2014), *PLoS Comp Bio* — RSA toolbox paper
- King & Dehaene (2014), *TiCS* — temporal generalization framework (already familiar; relevant for time-resolved analyses)
