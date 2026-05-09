"""
Encoding models for perceptXbind MEG analyses.

An encoding model predicts neural activity at each sensor/timepoint from a
trial-level feature representation:

    x_i(c, t) = W(t) @ phi(s_i) + noise

W is fit with ridge regression (RidgeCV for automatic lambda selection).
Evaluation is cross-validated R² or Pearson r per sensor per timepoint.

Feature spaces
--------------
stim_identity    : 4-d one-hot of current stimulus shape
stim_x_position  : 12-d one-hot of (shape, rule-phase) — tests position-binding
stim_x_role      : 8-d one-hot of (shape, role A/B)   — tests role-binding
rule_type        : 2-d one-hot of rule type (ABA / ABB)
combined         : horizontal stack of stim_identity + stim_x_role + rule_type
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from .cv import CVSplitter
from .conditions import (
    PHASE_OFFSETS, RULE_PHASES,
    _rule_phase_stim_role, STIM_CATEGORIES,
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EncodingResult:
    """Output of fit_encoding_model.

    scores        : (n_folds, n_channels, n_times) — R² or Pearson r
    weights       : (n_folds, n_channels, n_features, n_times) — optional
    alphas        : (n_folds, n_channels, n_times) — selected λ per fold
    feature_names : list of feature dimension names
    score_type    : 'r2' or 'pearson'
    """
    scores:        np.ndarray
    alphas:        np.ndarray
    feature_names: List[str]
    score_type:    str = 'r2'
    weights:       Optional[np.ndarray] = None

    @property
    def mean_scores(self) -> np.ndarray:
        """(n_channels, n_times) — mean CV score."""
        return self.scores.mean(axis=0)

    @property
    def sem_scores(self) -> np.ndarray:
        """(n_channels, n_times) — SEM across folds."""
        return self.scores.std(axis=0) / np.sqrt(self.scores.shape[0])

    def __repr__(self) -> str:
        n_folds, n_ch, n_t = self.scores.shape
        grand = self.mean_scores.mean()
        return (
            f"EncodingResult(score={self.score_type}, folds={n_folds}, "
            f"channels={n_ch}, times={n_t}, "
            f"features={self.feature_names}, grand_mean={grand:.4f})"
        )


# ── Feature space construction ────────────────────────────────────────────────

def build_feature_space(
    df: pd.DataFrame,
    epoch_phase: np.ndarray,
    feature_name: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a per-epoch feature matrix from behavioral metadata.

    Parameters
    ----------
    df           : behavioral DataFrame, 120 rows (one per trial)
    epoch_phase  : (1080,) phase name per epoch (from assign_conditions)
    feature_name : one of:
        'stim_identity'   — 4-d one-hot of shape
        'stim_x_position' — 12-d one-hot of (shape, rule-phase)
        'stim_x_role'     — 8-d one-hot of (shape, role)
        'rule_type'       — 2-d one-hot of rule type
        'combined'        — stim_identity + stim_x_role + rule_type (14-d)

    Returns
    -------
    Phi        : (n_epochs, n_features) float32 feature matrix; 0-filled
                 for epochs outside the scheme (fixation, transition, etc.)
    feat_names : list of feature dimension names (length n_features)
    """
    VALID = ('stim_identity', 'stim_x_position', 'stim_x_role', 'rule_type', 'combined')
    if feature_name not in VALID:
        raise ValueError(f"feature_name must be one of {VALID}")

    n_epochs = len(epoch_phase)
    n_trials = len(df)

    stim_cats = sorted(STIM_CATEGORIES)          # 4 shapes
    rule_types = sorted(df['rule_type'].unique()) # ['ABA', 'ABB']

    if feature_name == 'stim_identity':
        return _stim_identity_features(df, epoch_phase, n_trials, stim_cats)

    elif feature_name == 'stim_x_position':
        return _stim_x_position_features(df, epoch_phase, n_trials, stim_cats)

    elif feature_name == 'stim_x_role':
        return _stim_x_role_features(df, epoch_phase, n_trials, stim_cats)

    elif feature_name == 'rule_type':
        return _rule_type_features(df, epoch_phase, n_trials, rule_types)

    elif feature_name == 'combined':
        Phi_si, fn_si = _stim_identity_features(df, epoch_phase, n_trials, stim_cats)
        Phi_sr, fn_sr = _stim_x_role_features(df, epoch_phase, n_trials, stim_cats)
        Phi_rt, fn_rt = _rule_type_features(df, epoch_phase, n_trials, rule_types)
        return np.concatenate([Phi_si, Phi_sr, Phi_rt], axis=1), fn_si + fn_sr + fn_rt


def build_feature_space_nonepoched(
    df: pd.DataFrame,
    feature_name: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a per-epoch feature matrix from behavioral metadata.

    Parameters
    ----------
    df           : behavioral DataFrame, 120 rows (one per trial)
    feature_name : one of:
        'stim_identity'   — 4-d one-hot of shape
        'stim_x_position' — 12-d one-hot of (shape, rule-phase)
        'stim_x_role'     — 8-d one-hot of (shape, role)
        'rule_type'       — 2-d one-hot of rule type
        'combined'        — stim_identity + stim_x_role + rule_type (14-d)

    Returns
    -------
    Phi        : (n_epochs, n_features) float32 feature matrix; 0-filled
                 for epochs outside the scheme (fixation, transition, etc.)
    feat_names : list of feature dimension names (length n_features)
    """
    VALID = ('stim_A_identity', 'stim_B_identity', "stim_X_identity", "stim_Y_identity", 'stim_x_role', 'rule_type', 'combined')
    if feature_name not in VALID:
        raise ValueError(f"feature_name must be one of {VALID}")

    n_trials = len(df)

    stim_cats = sorted(STIM_CATEGORIES)          # 4 shapes
    rule_types = sorted(df['rule_type'].unique()) # ['ABA', 'ABB']

    if feature_name == 'stim_A_identity':
        return _stim_identity_features_ne(df, 'A', n_trials)
    elif feature_name == 'stim_B_identity':
        return _stim_identity_features_ne(df, 'B', n_trials)
    elif feature_name == 'stim_X_identity':
        return _stim_identity_features_ne(df, 'X', n_trials)
    elif feature_name == 'stim_Y_identity':
        return _stim_identity_features_ne(df, 'Y', n_trials)


    elif feature_name == 'rule_type':
        return _rule_type_features_ne(df, n_trials, rule_types)

    elif feature_name == 'combined':
        Phi_si, fn_si = _stim_identity_features_ne(df, 'A', n_trials)
        Phi_sr, fn_sr = _stim_identity_features_ne(df, 'B', n_trials)
        Phi_sx, fn_sx = _stim_identity_features_ne(df, 'X', n_trials)
        Phi_sy, fn_sy = _stim_identity_features_ne(df, 'Y', n_trials)
        Phi_rt, fn_rt = _rule_type_features_ne(df, n_trials, rule_types)
        return np.concatenate([Phi_si, Phi_sr, Phi_sx, Phi_sy, Phi_rt], axis=1), fn_si + fn_sr + fn_sx + fn_sy + fn_rt



def _one_hot(labels, vocab):
    """(n,) labels → (n, |vocab|) float32 one-hot matrix."""
    vocab_map = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(labels), len(vocab)), dtype=np.float32)
    for i, lbl in enumerate(labels):
        if lbl in vocab_map:
            out[i, vocab_map[lbl]] = 1.0
    return out


def _stim_identity_features(df, epoch_phase, n_trials, stim_cats):
    n_epochs = len(epoch_phase)
    Phi = np.zeros((n_epochs, len(stim_cats)), dtype=np.float32)
    names = [f'shape_{s}' for s in stim_cats]

    for phase in RULE_PHASES:
        offset = PHASE_OFFSETS[phase]
        stim, _ = _rule_phase_stim_role(df, phase)
        oh = _one_hot(stim, stim_cats)
        for t in range(n_trials):
            Phi[9 * t + offset] = oh[t]

    return Phi, names

def _stim_identity_features_ne(df, ep, n_trials):
    name = f'shape_{ep}'
    Phi = _one_hot(
        df[ep if ep in df.columns else f"{ep}_stim"].iloc[:n_trials].to_numpy(),
        sorted(STIM_CATEGORIES),
    ); names = [f"{name}_{s}" for s in sorted(STIM_CATEGORIES)]

    return Phi, names



def _stim_x_position_features(df, epoch_phase, n_trials, stim_cats):
    positions = RULE_PHASES  # ['rule1', 'rule2', 'rule3']
    vocab = [(s, p) for p in positions for s in stim_cats]  # 12
    names = [f'shape_{s}_pos_{p}' for s, p in vocab]
    vocab_map = {v: i for i, v in enumerate(vocab)}

    n_epochs = len(epoch_phase)
    Phi = np.zeros((n_epochs, len(vocab)), dtype=np.float32)

    for phase in RULE_PHASES:
        offset = PHASE_OFFSETS[phase]
        stim, _ = _rule_phase_stim_role(df, phase)
        for t in range(n_trials):
            key = (stim[t], phase)
            if key in vocab_map:
                Phi[9 * t + offset, vocab_map[key]] = 1.0

    return Phi, names


def _stim_x_role_features(df, epoch_phase, n_trials, stim_cats):
    roles = ['A', 'B']
    vocab = [(s, r) for r in roles for s in stim_cats]  # 8
    names = [f'shape_{s}_role_{r}' for s, r in vocab]
    vocab_map = {v: i for i, v in enumerate(vocab)}

    n_epochs = len(epoch_phase)
    Phi = np.zeros((n_epochs, len(vocab)), dtype=np.float32)

    for phase in RULE_PHASES:
        offset = PHASE_OFFSETS[phase]
        stim, role = _rule_phase_stim_role(df, phase)
        for t in range(n_trials):
            key = (stim[t], role[t])
            if key in vocab_map:
                Phi[9 * t + offset, vocab_map[key]] = 1.0

    return Phi, names


def _rule_type_features(df, epoch_phase, n_trials, rule_types):
    names = [f'rule_{rt}' for rt in rule_types]
    rt_map = {rt: i for i, rt in enumerate(rule_types)}

    n_epochs = len(epoch_phase)
    Phi = np.zeros((n_epochs, len(rule_types)), dtype=np.float32)

    for phase in RULE_PHASES:
        offset = PHASE_OFFSETS[phase]
        for t in range(n_trials):
            rt = df['rule_type'].iloc[t]
            Phi[9 * t + offset, rt_map[rt]] = 1.0

    return Phi, names

def _rule_type_features_ne(df, n_trials, rule_types):
    names = [f'rule_{rt}' for rt in rule_types]
    rt_map = {rt: i for i, rt in enumerate(rule_types)}

    n_epochs = len(df)
    Phi = np.zeros((n_epochs, len(rule_types)), dtype=np.float32)

    for t in range(n_trials):
        rt = df['rule_type'].iloc[t]
        if rt in rt_map:
            Phi[t, rt_map[rt]] = 1.0

    return Phi, names

# ── Encoding model fitting ────────────────────────────────────────────────────

def fit_encoding_model(
    X: np.ndarray,
    Phi: np.ndarray,
    cv: CVSplitter,
    feature_names: Optional[List[str]] = None,
    alpha_grid: Optional[np.ndarray] = None,
    score: str = 'r2',
    store_weights: bool = False,
) -> EncodingResult:
    """
    Fit a cross-validated ridge encoding model per sensor per timepoint.

    Parameters
    ----------
    X             : (n_epochs, n_channels, n_times) MEG data
    Phi           : (n_epochs, n_features) feature matrix
    cv            : CVSplitter for outer cross-validation
    feature_names : optional list of feature names (length n_features)
    alpha_grid    : regularization grid for RidgeCV; default log-spaced 1e-3..1e3
    score         : 'r2' | 'pearson'
    store_weights : if True, store weight matrices (large: n_folds×n_ch×n_feat×n_times)

    Returns
    -------
    EncodingResult
    """
    if score not in ('r2', 'pearson'):
        raise ValueError(f"score must be 'r2' or 'pearson', got {score!r}")

    n_epochs, n_ch, n_times = X.shape
    n_features = Phi.shape[1]

    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 20)

    if feature_names is None:
        feature_names = [f'feat_{i}' for i in range(n_features)]

    folds = list(cv.split(X, np.zeros(n_epochs)))  # y ignored, just need splits
    n_folds = len(folds)

    all_scores  = np.zeros((n_folds, n_ch, n_times), dtype=np.float32)
    all_alphas  = np.zeros((n_folds, n_ch, n_times), dtype=np.float32)
    all_weights = (
        np.zeros((n_folds, n_ch, n_features, n_times), dtype=np.float32)
        if store_weights else None
    )

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        Phi_train = Phi[train_idx]   # (n_tr, n_feat)
        Phi_test  = Phi[test_idx]    # (n_te, n_feat)
        X_train   = X[train_idx]     # (n_tr, n_ch, n_times)
        X_test    = X[test_idx]      # (n_te, n_ch, n_times)

        # Fit one RidgeCV per timepoint (vectorized over channels)
        for t in range(n_times):
            y_train = X_train[:, :, t]  # (n_tr, n_ch)
            y_test  = X_test[:, :, t]   # (n_te, n_ch)

            # RidgeCV fits all channels simultaneously (multi-output)
            ridge = RidgeCV(alphas=alpha_grid, fit_intercept=True)
            ridge.fit(Phi_train, y_train)

            y_pred = ridge.predict(Phi_test)  # (n_te, n_ch)

            if score == 'r2':
                for ci in range(n_ch):
                    ss_res = ((y_test[:, ci] - y_pred[:, ci]) ** 2).sum()
                    ss_tot = ((y_test[:, ci] - y_test[:, ci].mean()) ** 2).sum()
                    all_scores[fold_idx, ci, t] = (
                        1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                    )
            else:  # pearson
                for ci in range(n_ch):
                    if y_test[:, ci].std() < 1e-12 or y_pred[:, ci].std() < 1e-12:
                        all_scores[fold_idx, ci, t] = 0.0
                    else:
                        r = np.corrcoef(y_test[:, ci], y_pred[:, ci])[0, 1]
                        all_scores[fold_idx, ci, t] = float(r) if np.isfinite(r) else 0.0

            # Store selected alpha (same for all channels in RidgeCV multi-output)
            all_alphas[fold_idx, :, t] = ridge.alpha_

            if store_weights:
                # ridge.coef_ shape: (n_ch, n_features) for multi-output
                all_weights[fold_idx, :, :, t] = ridge.coef_.astype(np.float32)

    return EncodingResult(
        scores=all_scores,
        alphas=all_alphas,
        feature_names=list(feature_names),
        score_type=score,
        weights=all_weights,
    )


# ── Nested model comparison ───────────────────────────────────────────────────

def nested_comparison(
    result_A: EncodingResult,
    result_B: EncodingResult,
) -> np.ndarray:
    """
    Compute unique variance of model B beyond model A.

    Returns (n_folds, n_channels, n_times) difference scores R²_B − R²_A.
    Positive values indicate that B's additional features improve prediction.
    """
    return result_B.scores - result_A.scores
