"""
Representational Similarity Analysis (RSA) for MEG decoding toolkit.

Pipeline
--------
1. Compute neural RDM from condition-mean patterns (crossnobis or correlation)
2. Build model RDMs from condition metadata
3. Compare neural RDMs to model RDMs (Spearman or partial regression)
4. Output RSAResult with time-resolved fits

Crossnobis implementation
-------------------------
Uses leave-one-CV-fold-out cross-validation: the distance between two
conditions is estimated as the inner product of the pattern difference
computed from training data with the pattern difference computed from the
held-out test fold. This estimator is unbiased under the null (expected 0
when conditions have identical distributions).

References
----------
Diedrichsen & Kriegeskorte (2017), PLoS Comp Bio — crossnobis & RSA
Nili et al. (2014), PLoS Comp Bio — RSA toolbox
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from .cv import CVSplitter


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RSAResult:
    """Output of time_resolved_rsa.

    fits        : (n_models, n_times)  Spearman ρ or regression β per model
    model_names : list of model names aligned with fits axis 0
    times       : (n_times,) time vector in seconds
    neural_rdms : (n_times, K, K) neural RDMs (optional, large; set store_rdms=True)
    method      : 'spearman' | 'regression'
    distance    : 'crossnobis' | 'correlation'
    """
    fits:        np.ndarray
    model_names: List[str]
    times:       np.ndarray
    neural_rdms: Optional[np.ndarray] = None
    method:      str = 'spearman'
    distance:    str = 'crossnobis'

    @property
    def mean_fits(self) -> np.ndarray:
        return self.fits  # already averaged; kept for API symmetry

    def __repr__(self) -> str:
        return (
            f"RSAResult(models={self.model_names}, "
            f"n_times={len(self.times)}, method={self.method!r}, "
            f"distance={self.distance!r})"
        )


# ── Model RDM construction ────────────────────────────────────────────────────

def build_model_rdm(
    condition_metadata: pd.DataFrame,
    model_name: str,
) -> np.ndarray:
    """
    Build a (K, K) model RDM from condition metadata.

    Parameters
    ----------
    condition_metadata : DataFrame with one row per condition; must contain
                         relevant columns depending on model_name.
    model_name : one of:
        'stimulus_identity'  — 0 if same shape, 1 if different
        'abstract_role'      — 0 if same role (A/B), 1 if different
        'rule_type'          — 0 if same rule_type, 1 if different
        'conjunctive_binding'— 0 if same shape AND role, 1 otherwise
        'position'           — 0 if same phase, 1 if different

    Returns
    -------
    (K, K) float64, zero diagonal, symmetric.
    """
    K = len(condition_metadata)
    rdm = np.zeros((K, K), dtype=np.float64)

    meta = condition_metadata.reset_index(drop=True)

    def _binary(arr):
        for i in range(K):
            for j in range(K):
                rdm[i, j] = 0.0 if arr[i] == arr[j] else 1.0
        np.fill_diagonal(rdm, 0.0)

    if model_name == 'stimulus_identity':
        if 'shape' not in meta.columns:
            raise ValueError("condition_metadata must have 'shape' column for stimulus_identity")
        _binary(meta['shape'].values)
    
    elif model_name == 'stimulus_A_identity':
        # if 'shape' not in meta.columns or 'role' not in meta.columns:
            # raise ValueError("condition_metadata must have 'shape' and 'role' columns for stimulus_A_identity")
        _binary(meta['A_shape'].values)
    
    elif model_name == 'stimulus_B_identity':
        # if 'shape' not in meta.columns or 'role' not in meta.columns:
            # raise ValueError("condition_metadata must have 'shape' and 'role' columns for stimulus_B_identity")
        _binary(meta['B_shape'].values)
    
    elif model_name == 'stimulus_X_identity':
        _binary(meta['X_shape'].values)
    
    elif model_name == 'stimulus_Y_identity':
        _binary(meta['Y_shape'].values)
    
    elif model_name == 'abstract_role':
        if 'role' not in meta.columns:
            raise ValueError("condition_metadata must have 'role' column for abstract_role")
        _binary(meta['role'].values)

    elif model_name == 'rule_type':
        if 'rule_type' not in meta.columns:
            raise ValueError("condition_metadata must have 'rule_type' column for rule_type")
        _binary(meta['rule_type'].values)

    elif model_name == 'conjunctive_binding':
        if 'shape' not in meta.columns or 'role' not in meta.columns:
            raise ValueError("condition_metadata needs 'shape' and 'role' for conjunctive_binding")
        for i in range(K):
            for j in range(K):
                same = (meta['shape'].iloc[i] == meta['shape'].iloc[j] and
                        meta['role'].iloc[i] == meta['role'].iloc[j])
                rdm[i, j] = 0.0 if same else 1.0
        np.fill_diagonal(rdm, 0.0)

    elif model_name == 'position':
        if 'phase' not in meta.columns:
            raise ValueError("condition_metadata must have 'phase' column for position")
        _binary(meta['phase'].values)

    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    return rdm


def check_model_rdm_collinearity(
    model_rdms: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute VIF for all model RDM vectors (upper triangle).

    Returns DataFrame with columns [model, vif].
    Warns if any VIF > 5.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    names = list(model_rdms.keys())
    K = next(iter(model_rdms.values())).shape[0]
    triu = np.triu_indices(K, k=1)

    X = np.column_stack([rdm[triu] for rdm in model_rdms.values()])
    # Add small jitter to avoid exact collinearity crashing VIF
    X = X + 1e-10 * np.random.default_rng(0).standard_normal(X.shape)

    vifs = []
    for i in range(X.shape[1]):
        try:
            vif = variance_inflation_factor(X, i)
        except Exception:
            vif = np.nan
        vifs.append(vif)

    df = pd.DataFrame({'model': names, 'vif': vifs})
    high = df[df['vif'] > 5]
    if len(high) > 0:
        import warnings
        warnings.warn(
            f"High VIF (>5) detected for: {high['model'].tolist()}. "
            "Consider dropping collinear models before regression.",
            stacklevel=2,
        )
    return df


# ── Neural RDM computation ────────────────────────────────────────────────────

def _correlation_rdm_series(
    X: np.ndarray,
    condition_ids: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """
    Pearson correlation distance between condition-mean patterns.

    For each timepoint and condition pair (c1, c2):
        d = 1 - r(mean_c1, mean_c2)
    where r is the Pearson correlation (mean-centered then L2-normalized).

    Returns (n_times, K, K).
    """
    n_epochs, n_ch, n_times = X.shape
    K = len(classes)

    # Condition means: (K, n_ch, n_times)
    means = np.zeros((K, n_ch, n_times), dtype=np.float64)
    for ci, c in enumerate(classes):
        idx = condition_ids == c
        means[ci] = X[idx].mean(axis=0)

    rdm = np.zeros((n_times, K, K), dtype=np.float64)
    for t in range(n_times):
        patterns = means[:, :, t]  # (K, n_ch)
        # Pearson: mean-center each pattern across channels, then L2-normalize
        centered = patterns - patterns.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normed = centered / norms
        corr_mat = normed @ normed.T   # (K, K) — Pearson r matrix
        corr_mat = np.clip(corr_mat, -1.0, 1.0)
        rdm[t] = 1.0 - corr_mat
        np.fill_diagonal(rdm[t], 0.0)

    return rdm


def _crossnobis_rdm_series(
    X: np.ndarray,
    condition_ids: np.ndarray,
    classes: np.ndarray,
    cv: CVSplitter,
) -> np.ndarray:
    """
    Crossnobis (cross-validated Mahalanobis) distance.

    For each CV fold:
        d_fold[c1,c2,t] = (Δ_train)ᵀ Σ_t⁻¹ (Δ_test)

    where Δ = mean_c1 − mean_c2 and Σ_t is the pooled within-condition
    residual covariance estimated on the training set at timepoint t.

    Σ is regularized as Σ_reg = (1-α)Σ + α·tr(Σ)/p · I  (Ledoit-Wolf-style)
    with a fixed shrinkage of 0.1, making the estimator robust when n < p.

    Estimator is unbiased: E[d(c1,c2)] = 0 when conditions are identical.
    Distances can be negative — that is a feature, not a bug.

    Returns (n_times, K, K).
    """
    n_epochs, n_ch, n_times = X.shape
    K = len(classes)
    SHRINKAGE = 0.1   # fixed Ledoit-Wolf-style regularization

    folds = list(cv.split(X, condition_ids))
    n_folds = len(folds)

    rdm_accum = np.zeros((n_times, K, K), dtype=np.float64)

    for train_idx, test_idx in folds:
        y_train = condition_ids[train_idx]
        y_test  = condition_ids[test_idx]

        # Condition means: (K, n_ch, n_times)
        train_means = np.zeros((K, n_ch, n_times), dtype=np.float64)
        test_means  = np.zeros((K, n_ch, n_times), dtype=np.float64)
        for ci, c in enumerate(classes):
            tr_mask = y_train == c
            te_mask = y_test  == c
            if tr_mask.sum() > 0:
                train_means[ci] = X[train_idx[tr_mask]].mean(axis=0)
            if te_mask.sum() > 0:
                test_means[ci] = X[test_idx[te_mask]].mean(axis=0)

        # Pooled within-condition residuals: (n_tr, n_ch, n_times)
        n_tr = len(train_idx)
        residuals = np.zeros((n_tr, n_ch, n_times), dtype=np.float64)
        for ci, c in enumerate(classes):
            mask = y_train == c
            if mask.sum() > 0:
                residuals[mask] = X[train_idx[mask]] - train_means[ci]

        # Precompute all pairwise differences: (n_pairs, n_ch, n_times)
        pairs = [(ci, cj) for ci in range(K) for cj in range(ci + 1, K)]
        diff_tr_all = np.stack(
            [train_means[ci] - train_means[cj] for ci, cj in pairs], axis=0
        )  # (n_pairs, n_ch, n_times)
        diff_te_all = np.stack(
            [test_means[ci] - test_means[cj] for ci, cj in pairs], axis=0
        )

        # One Sigma per timepoint; apply to all pairs at once
        dof = max(n_tr - K, 1)
        eye = np.eye(n_ch)

        for t in range(n_times):
            r = residuals[:, :, t]                    # (n_tr, n_ch)
            Sigma = (r.T @ r) / dof                   # (n_ch, n_ch)
            tr_S = np.trace(Sigma)
            Sigma_reg = (1 - SHRINKAGE) * Sigma + SHRINKAGE * (tr_S / n_ch) * eye

            # Solve Sigma_reg @ V = diff_tr_all[:, :, t].T  → V: (n_ch, n_pairs)
            try:
                V = np.linalg.solve(Sigma_reg, diff_tr_all[:, :, t].T)  # (n_ch, n_pairs)
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(Sigma_reg, diff_tr_all[:, :, t].T, rcond=None)[0]

            # d[pair] = diff_te[:, t] · V[:, pair]
            d = (diff_te_all[:, :, t] * V.T).sum(axis=1)  # (n_pairs,)

            for p, (ci, cj) in enumerate(pairs):
                rdm_accum[t, ci, cj] += d[p]
                rdm_accum[t, cj, ci] += d[p]

    rdm_accum /= n_folds
    return rdm_accum


# ── RDM comparison ────────────────────────────────────────────────────────────

def _spearman_comparison(
    neural_rdm: np.ndarray,
    model_rdms: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Spearman correlation between neural and each model RDM, time-resolved.

    neural_rdm : (n_times, K, K)
    Returns    : {model_name: (n_times,) Spearman ρ}
    """
    K = neural_rdm.shape[1]
    triu = np.triu_indices(K, k=1)
    n_times = neural_rdm.shape[0]

    result = {}
    for name, mrdm in model_rdms.items():
        m_vec = mrdm[triu]
        rhos = np.zeros(n_times)
        for t in range(n_times):
            n_vec = neural_rdm[t][triu]
            rho, _ = spearmanr(n_vec, m_vec)
            rhos[t] = rho if np.isfinite(rho) else 0.0
        result[name] = rhos

    return result


def _regression_comparison(
    neural_rdm: np.ndarray,
    model_rdms: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Partial RDM-space regression: fit neural distances ~ sum(beta_m * model_m).

    neural_rdm : (n_times, K, K)
    Returns    : {model_name: (n_times,) regression β}
    """
    from sklearn.linear_model import LinearRegression

    K = neural_rdm.shape[1]
    triu = np.triu_indices(K, k=1)
    n_times = neural_rdm.shape[0]
    names = list(model_rdms.keys())

    # Model RDM matrix: (n_pairs, n_models) — fixed across time
    X_models = np.column_stack([rdm[triu] for rdm in model_rdms.values()])

    betas = {name: np.zeros(n_times) for name in names}

    for t in range(n_times):
        y = neural_rdm[t][triu].astype(np.float64)
        # Rank the neural distances (Spearman-style Y, raw X for binary models)
        y_ranked = rankdata(y).astype(np.float64)
        reg = LinearRegression(fit_intercept=True).fit(X_models, y_ranked)
        for i, name in enumerate(names):
            betas[name][t] = reg.coef_[i]

    return betas


# ── Main function ─────────────────────────────────────────────────────────────

def time_resolved_rsa(
    X: np.ndarray,
    condition_ids: np.ndarray,
    model_rdms: Dict[str, np.ndarray],
    times: np.ndarray,
    method: str = 'regression',
    distance: str = 'crossnobis',
    cv: Optional[CVSplitter] = None,
    store_rdms: bool = False,
) -> RSAResult:
    """
    Time-resolved RSA: compute neural RDMs and compare to model RDMs.

    Parameters
    ----------
    X             : (n_epochs, n_channels, n_times) MEG data
    condition_ids : (n_epochs,) integer condition label; must not contain -1
    model_rdms    : {name: (K, K)} model RDMs, K = n unique conditions
    times         : (n_times,) time axis in seconds
    method        : 'spearman' | 'regression'
    distance      : 'crossnobis' | 'correlation'
    cv            : CVSplitter (required for crossnobis; default 5-fold stratified)
    store_rdms    : if True, store (n_times, K, K) neural RDMs in result

    Returns
    -------
    RSAResult
    """
    if -1 in condition_ids:
        raise ValueError(
            "condition_ids contains -1 (out-of-scheme epochs). "
            "Filter to relevant epochs before calling time_resolved_rsa."
        )
    if method not in ('spearman', 'regression'):
        raise ValueError(f"method must be 'spearman' or 'regression', got {method!r}")
    if distance not in ('crossnobis', 'correlation'):
        raise ValueError(f"distance must be 'crossnobis' or 'correlation', got {distance!r}")

    classes = np.unique(condition_ids)
    K = len(classes)

    # Remap condition_ids to dense 0..K-1 in case of gaps
    remap = {c: i for i, c in enumerate(classes)}
    cids_dense = np.array([remap[c] for c in condition_ids], dtype=int)

    # ── Compute neural RDM series ─────────────────────────────────────────────
    if distance == 'crossnobis':
        if cv is None:
            cv = CVSplitter(n_splits=5, stratified=True, shuffle=True, random_state=42)
        neural_rdms_series = _crossnobis_rdm_series(X, cids_dense, np.arange(K), cv)
    else:
        neural_rdms_series = _correlation_rdm_series(X, cids_dense, np.arange(K))

    # ── Compare to model RDMs ─────────────────────────────────────────────────
    if method == 'spearman':
        fits_dict = _spearman_comparison(neural_rdms_series, model_rdms)
    else:
        fits_dict = _regression_comparison(neural_rdms_series, model_rdms)

    model_names = list(model_rdms.keys())
    fits = np.stack([fits_dict[n] for n in model_names], axis=0)  # (n_models, n_times)

    return RSAResult(
        fits=fits,
        model_names=model_names,
        times=times,
        neural_rdms=neural_rdms_series if store_rdms else None,
        method=method,
        distance=distance,
    )


# ── Cross-subject / cross-modal convenience ────────────────────────────────────

def average_rdms(rdm_series_list: List[np.ndarray]) -> np.ndarray:
    """Average a list of (n_times, K, K) neural RDM series across subjects."""
    return np.stack(rdm_series_list, axis=0).mean(axis=0)
