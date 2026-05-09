# Pseudo-Trial Averaging — Implementation Note

## Context

We want to add pseudo-trial averaging to the decoding toolkit as an SNR-boosting preprocessing step. The goal is to compare two construction strategies empirically and pick whichever performs better on our data:

1. **Disjoint partitioning** — each original trial belongs to exactly one pseudo-trial.
2. **Bootstrap resampling** — pseudo-trials are sampled from original trials, with original trials potentially appearing in multiple pseudo-trials.

Both should be implemented as a single configurable function with a `mode` flag. The CV machinery should be able to call either one transparently.

## Mathematical summary (for context)

A pseudo-trial is the average of `k` original trials of the same class:

$$\bar{\mathbf{x}}_g = \frac{1}{k}\sum_{i \in g} \mathbf{x}_i$$

This boosts per-sample SNR by a factor of $\sqrt{k}$ (amplitude) or $k$ (variance). The two modes differ in how the groups `g` are constructed:

- **Disjoint**: partition the $N_c$ trials of class $c$ into $\lfloor N_c/k \rfloor$ non-overlapping groups. Pseudo-trials are independent.
- **Bootstrap**: generate $M$ pseudo-trials, each formed by sampling $k$ trials uniformly without replacement from the class. Pseudo-trials may share original trials, so they are correlated.

The effective sample size is bounded by $N_c/k$ in both cases — bootstrapping does not generate new information beyond what the original trials contain, even if $M$ is large. This is worth keeping in mind when interpreting results, but it does not affect implementation.

## Function signature

Implement in `toolkit/pseudo_trials.py`:

```python
def make_pseudo_trials(
    X: np.ndarray,                 # (n_trials, n_channels, n_times)
    y: np.ndarray,                 # (n_trials,) integer labels
    k: int = 5,                    # trials per pseudo-trial
    mode: str = "disjoint",        # "disjoint" | "bootstrap"
    n_pseudo_per_class: Optional[int] = None,
                                   # bootstrap: M; disjoint: capped at N_c // k
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct pseudo-trials by averaging within-class groups of original trials.

    Returns
    -------
    Xp : (n_pseudo, n_channels, n_times) averaged pseudo-trials
    yp : (n_pseudo,) class labels for pseudo-trials
    """
```

### Disjoint mode

For each class:
1. Shuffle the within-class trial indices using `rng`.
2. Take the first $\lfloor N_c / k \rfloor \cdot k$ of them and reshape into groups of size $k$.
3. Average within each group.
4. If `n_pseudo_per_class` is set and is *less* than $\lfloor N_c/k \rfloor$, truncate. If set *higher*, raise an error — disjoint cannot produce more pseudo-trials than $N_c/k$ without reuse.

Trailing trials that don't fill a group are dropped. This is a small data loss but keeps the construction clean.

### Bootstrap mode

For each class:
1. Use `rng` to draw `n_pseudo_per_class` groups, each of size `k`, from within-class trial indices.
2. Sampling within a group is **without replacement** (no original trial appears twice in the same pseudo-trial).
3. Sampling **across** groups is unconstrained (the same trial can appear in multiple pseudo-trials).
4. Average within each group.

If `n_pseudo_per_class` is `None` in bootstrap mode, default to $N_c / k$ (matching the disjoint count for fair comparison).

## CRITICAL: leakage prevention

**This is the part where it's easy to introduce subtle bugs that corrupt the entire analysis. Read this section carefully.**

The cardinal rule: **no original trial may contribute to both the training and test sets in any CV fold.**

### What goes wrong

Naively bootstrapping or disjoint-partitioning the entire dataset *before* splitting into CV folds will contaminate the test set. Original trial 17 may end up in pseudo-trial 5 (training) and pseudo-trial 134 (test). The decoder has been trained on a partial average of trial 17, and is now being tested on another partial average of trial 17 — that's leakage, and it will inflate test scores in a way that's nearly impossible to detect post-hoc.

```python
# WRONG — DO NOT DO THIS
Xp, yp = make_pseudo_trials(X, y, k=5, mode="bootstrap", n_pseudo_per_class=100)
for train_idx, test_idx in cv.split(Xp, yp):
    decoder.fit(Xp[train_idx], yp[train_idx])
    score = decoder.score(Xp[test_idx], yp[test_idx])
```

### What to do instead

The CV split must happen on **original trials**, and pseudo-trials must be constructed independently within each fold's training and test sets:

```python
# CORRECT
for train_idx, test_idx in cv.split(X, y):
    Xp_train, yp_train = make_pseudo_trials(
        X[train_idx], y[train_idx], k=5, mode=mode, rng=rng
    )
    Xp_test, yp_test = make_pseudo_trials(
        X[test_idx], y[test_idx], k=5, mode=mode, rng=rng
    )
    # Or: test on single trials, see "asymmetric option" below
    decoder.fit(Xp_train, yp_train)
    score = decoder.score(Xp_test, yp_test)
```

### The asymmetric option (often preferred)

For our data, test folds are small ($\sim$24 trials per fold with 5-fold CV), so pseudo-trial averaging on the test side leaves very few test samples per class. Often it's better to train on pseudo-trials but evaluate on single trials:

```python
for train_idx, test_idx in cv.split(X, y):
    Xp_train, yp_train = make_pseudo_trials(
        X[train_idx], y[train_idx], k=5, mode=mode, rng=rng
    )
    decoder.fit(Xp_train, yp_train)
    score = decoder.score(X[test_idx], y[test_idx])  # single trials
```

The decoder trains on a higher-SNR signal but is evaluated on the realistic single-trial generalization. This is often what we want to report.

Make `cross_validate` accept a flag like `pseudo_test: bool = False` controlling whether pseudo-trial averaging applies to the test side or only the train side.

### Multiple repetitions for variance reduction

Both disjoint and bootstrap construction depend on a random seed. To reduce variance from "unlucky" groupings, repeat the procedure with `R` different seeds and average the test scores:

```python
fold_scores_per_rep = []
for rep in range(R):
    rng_rep = np.random.default_rng(seed=base_seed + rep)
    Xp_train, yp_train = make_pseudo_trials(
        X[train_idx], y[train_idx], k=5, mode=mode, rng=rng_rep
    )
    decoder = decoder_factory()
    decoder.fit(Xp_train, yp_train)
    fold_scores_per_rep.append(decoder.score(X[test_idx], y[test_idx]))
fold_score = np.mean(fold_scores_per_rep)
```

Use `R=20` as a default. This applies to both modes equally.

### Marginalization within pseudo-trials

For our stimulus identity decoders, we are explicitly marginalizing over nuisance variables (B, X, Y, rule type). The pseudo-trial sampling within a class can be:

- **Unstratified**: just sample uniformly from all trials with the target label. Marginalizes naturally if the design is balanced.
- **Stratified**: optionally enforce that each pseudo-trial samples evenly across nuisance levels (e.g., evenly across rule types).

For v1, implement **unstratified** averaging only. We can add stratification later if we need it for specific analyses (e.g., partial marginalization stratifying on rule type for late-trial decoders).

## Integration into `cross_validate`

Extend the existing function with the following parameters:

```python
def cross_validate(
    feature: Feature,
    decoder_factory: Callable[[], Decoder],
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    cv: CVSplitter,
    metric: str = "accuracy",
    return_predictions: bool = False,
    return_train_scores: bool = False,
    # NEW:
    pseudo_k: int = 1,                    # 1 = no averaging
    pseudo_mode: str = "disjoint",        # "disjoint" | "bootstrap"
    pseudo_n_per_class: Optional[int] = None,
    pseudo_test: bool = False,            # average on test side too?
    pseudo_repetitions: int = 1,          # how many random groupings
    pseudo_seed: int = 42,
) -> CVResult:
```

When `pseudo_k == 1`, behaviour is unchanged (current pipeline). When `pseudo_k > 1`, the loop becomes:

```python
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    rep_scores = []
    for rep in range(pseudo_repetitions):
        rng = np.random.default_rng(seed=pseudo_seed + 1000 * fold_idx + rep)

        # Pseudo-trials on train side
        X_train_proc, y_train_proc = make_pseudo_trials(
            X[train_idx], y[train_idx],
            k=pseudo_k, mode=pseudo_mode,
            n_pseudo_per_class=pseudo_n_per_class, rng=rng,
        )

        # Pseudo-trials on test side (optional)
        if pseudo_test:
            X_test_proc, y_test_proc = make_pseudo_trials(
                X[test_idx], y[test_idx],
                k=pseudo_k, mode=pseudo_mode,
                n_pseudo_per_class=pseudo_n_per_class, rng=rng,
            )
        else:
            X_test_proc, y_test_proc = X[test_idx], y[test_idx]

        # Fresh feature, fit on train pseudo-trials only
        feature_fold = feature.clone()
        feature_fold.fit(X_train_proc, sfreq, times)
        f_train = feature_fold.transform(X_train_proc)
        f_test  = feature_fold.transform(X_test_proc)

        # Fresh decoder
        decoder = decoder_factory()
        decoder.fit(f_train, y_train_proc)

        rep_scores.append(decoder.score(f_test, y_test_proc, metric=metric))

    # Average across repetitions
    fold_score = np.mean(np.stack(rep_scores), axis=0)
    all_test_scores.append(fold_score)
```

Note that the feature fits on the pseudo-trials, not the originals. This is the cleanest behaviour because feature standardization (z-scoring) should reflect the actual training distribution the decoder sees.

## Tests to write

Add these to the existing test suite:

1. **Shape correctness**: `make_pseudo_trials` returns the expected number of pseudo-trials per class for both modes.
2. **Disjoint coverage**: in disjoint mode, each original trial appears in exactly one pseudo-trial (verify with an `id` accumulator).
3. **No within-pseudo-trial duplicates**: in either mode, the trials averaged into a single pseudo-trial are distinct.
4. **No CV leakage**: construct a synthetic dataset where each trial has a unique signature in one channel; run `cross_validate` with `pseudo_k > 1` for both modes; verify that the train/test split, traced back to original trials, never reuses an original trial. Add this as a regression test — it's the most important one.
5. **SNR sanity check**: on synthetic data with a known signal-to-noise ratio, verify that pseudo-trial averaging with $k=5$ improves decoding accuracy compared to single-trial in a finite-trial regime.
6. **Determinism**: same seed gives identical pseudo-trials.

## Running the empirical comparison

Once both modes are implemented, run them on our existing rule3 stimulus decoding pipeline (`A_stim` decoding from rule1 epochs, since this is where marginalization is cleanest). Compare:

- Single-trial baseline (`pseudo_k=1`)
- Disjoint with $k \in \{3, 5\}$, R=20 repetitions
- Bootstrap with $k \in \{3, 5\}$, $M$ matching the disjoint count, R=20 repetitions
- Bootstrap with $k=5$, $M$ inflated to $4 \cdot N_c/k$ (to test whether more pseudo-trials per repetition helps)

Plot mean test AUC across folds and time, with shaded SEM across the R repetitions, for each configuration. The headline comparison is disjoint vs. bootstrap at matched $M$, and whether bootstrap with inflated $M$ adds anything.

Output the comparison as a single notebook (`examples/pseudo_trial_comparison.ipynb`).

## Summary of points to be careful about

1. **Always construct pseudo-trials *inside* the CV fold loop, not before.** This is the single most important rule.
2. **Use a fresh `Feature.clone()` and `decoder_factory()` per repetition.** No state should leak across reps.
3. **Track random seeds explicitly.** Each fold and each repetition should use a deterministic seed derived from a base seed, so results are reproducible.
4. **Don't claim independent test points equal to the bootstrap M.** When reporting standard errors or confidence intervals, the unit of independence is original trials, not pseudo-trials. Use the `pseudo_repetitions` axis for variance estimation, and report group-level statistics across subjects (when we get to multi-subject analysis) as the actual basis for inference.
