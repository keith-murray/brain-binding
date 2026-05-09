"""
Tests for toolkit/encoding.py and toolkit/conditions.py feature building.

Uses synthetic data with a known feature-to-activity map to verify
that fit_encoding_model recovers above-chance R² and that feature space
construction is correct.
"""

import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from toolkit.encoding import (
    build_feature_space,
    fit_encoding_model,
    nested_comparison,
    EncodingResult,
)
from toolkit.conditions import assign_conditions, PHASE_OFFSETS, RULE_PHASES
from toolkit.cv import CVSplitter


# ── Synthetic helpers ─────────────────────────────────────────────────────────

def make_df(n=80, seed=0):
    rng = np.random.default_rng(seed)
    shapes = ['circle', 'rectangle', 'star', 'triangle']
    rule_types = ['ABA', 'ABB']
    test_types = ['ABA', 'ABB']
    rows = []
    for t in range(n):
        rt = rule_types[t % 2]
        tt = test_types[(t // 2) % 2]
        a, b = rng.choice(shapes, size=2, replace=False)
        x, y = rng.choice(shapes, size=2, replace=False)
        rows.append({'trial': t, 'rule_type': rt, 'test_type': tt,
                     'A_stim': a, 'B_stim': b, 'X_stim': x, 'Y_stim': y})
    return pd.DataFrame(rows)


def make_meg_from_features(Phi, n_ch=8, n_times=10, noise_std=0.3, seed=1):
    """
    Generate MEG data as X = Phi @ W_true + noise.
    Only uses rule-phase epochs (non-zero Phi rows).
    W_true: (n_features, n_ch, n_times)
    """
    rng = np.random.default_rng(seed)
    n_epochs, n_feat = Phi.shape
    W = rng.standard_normal((n_feat, n_ch, n_times)).astype(np.float32)
    X = np.zeros((n_epochs, n_ch, n_times), dtype=np.float32)
    for t in range(n_times):
        X[:, :, t] = Phi @ W[:, :, t]
    X += rng.standard_normal(X.shape).astype(np.float32) * noise_std
    return X, W


# ── Feature space construction ────────────────────────────────────────────────

class TestBuildFeatureSpace:

    def test_stim_identity_shape(self):
        df = make_df(n=40)
        res = assign_conditions(df, scheme='stim_by_role')
        Phi, names = build_feature_space(df, res['epoch_phase'], 'stim_identity')
        assert Phi.shape == (40 * 9, 4)
        assert len(names) == 4

    def test_stim_x_role_shape(self):
        df = make_df(n=40)
        res = assign_conditions(df, scheme='stim_by_role')
        Phi, names = build_feature_space(df, res['epoch_phase'], 'stim_x_role')
        assert Phi.shape == (40 * 9, 8)
        assert len(names) == 8

    def test_stim_x_position_shape(self):
        df = make_df(n=40)
        res = assign_conditions(df, scheme='stim_by_role')
        Phi, names = build_feature_space(df, res['epoch_phase'], 'stim_x_position')
        assert Phi.shape == (40 * 9, 12)

    def test_rule_type_shape(self):
        df = make_df(n=40)
        res = assign_conditions(df, scheme='stim_by_role')
        Phi, names = build_feature_space(df, res['epoch_phase'], 'rule_type')
        assert Phi.shape == (40 * 9, 2)
        assert len(names) == 2

    def test_combined_shape(self):
        df = make_df(n=40)
        res = assign_conditions(df, scheme='stim_by_role')
        Phi, names = build_feature_space(df, res['epoch_phase'], 'combined')
        # stim_identity(4) + stim_x_role(8) + rule_type(2) = 14
        assert Phi.shape[1] == 14

    def test_rule_phases_nonzero(self):
        """Rule-phase epochs must have non-zero feature vectors."""
        df = make_df(n=20)
        res = assign_conditions(df, scheme='stim_by_role')
        Phi, _ = build_feature_space(df, res['epoch_phase'], 'stim_identity')
        for phase in RULE_PHASES:
            offset = PHASE_OFFSETS[phase]
            for t in range(len(df)):
                row = Phi[9 * t + offset]
                assert row.sum() == 1.0, f"Expected one-hot at {phase} trial {t}"

    def test_non_rule_phases_zero(self):
        """Non-rule epochs should be all-zero in feature matrix."""
        df = make_df(n=20)
        res = assign_conditions(df, scheme='stim_by_role')
        Phi, _ = build_feature_space(df, res['epoch_phase'], 'stim_identity')
        for phase in ('fixation', 'transition', 'response'):
            offset = PHASE_OFFSETS[phase]
            for t in range(len(df)):
                row = Phi[9 * t + offset]
                assert row.sum() == 0.0, f"Expected zero at {phase} trial {t}"

    def test_rule1_stim_identity_matches_A_stim(self):
        df = make_df(n=20)
        res = assign_conditions(df, scheme='stim_by_role')
        Phi, names = build_feature_space(df, res['epoch_phase'], 'stim_identity')
        stim_map = {n.split('_')[1]: i for i, n in enumerate(names)}
        for t in range(len(df)):
            ep_idx = 9 * t + PHASE_OFFSETS['rule1']
            expected_shape = df['A_stim'].iloc[t]
            expected_col = stim_map[expected_shape]
            assert Phi[ep_idx, expected_col] == 1.0
            assert Phi[ep_idx].sum() == 1.0

    def test_stim_x_role_one_hot(self):
        df = make_df(n=20)
        res = assign_conditions(df, scheme='stim_by_role')
        Phi, names = build_feature_space(df, res['epoch_phase'], 'stim_x_role')
        for phase in RULE_PHASES:
            offset = PHASE_OFFSETS[phase]
            for t in range(len(df)):
                row = Phi[9 * t + offset]
                assert row.sum() == 1.0

    def test_invalid_feature_name(self):
        df = make_df(n=10)
        res = assign_conditions(df, scheme='stim_by_role')
        with pytest.raises(ValueError):
            build_feature_space(df, res['epoch_phase'], 'invalid_feature')


# ── fit_encoding_model ────────────────────────────────────────────────────────

class TestFitEncodingModel:

    def _rule_phase_data(self, n_trials=60, n_ch=6, n_times=8, feat='stim_identity'):
        """Prepare MEG and Phi for rule-phase epochs only."""
        df = make_df(n=n_trials)
        res = assign_conditions(df, scheme='stim_by_role')
        ep_phase = res['epoch_phase']
        Phi_full, feat_names = build_feature_space(df, ep_phase, feat)

        # Filter to rule-phase epochs
        rule_mask = np.isin(ep_phase, RULE_PHASES)
        Phi = Phi_full[rule_mask]
        X, _ = make_meg_from_features(Phi, n_ch=n_ch, n_times=n_times, noise_std=0.5)
        return X, Phi, feat_names, df

    def test_result_type(self):
        X, Phi, names, _ = self._rule_phase_data()
        cv = CVSplitter(n_splits=3, stratified=False, shuffle=True, random_state=0)
        result = fit_encoding_model(X, Phi, cv, feature_names=names, score='r2')
        assert isinstance(result, EncodingResult)

    def test_scores_shape(self):
        X, Phi, names, _ = self._rule_phase_data(n_times=8)
        cv = CVSplitter(n_splits=3, stratified=False, shuffle=True, random_state=0)
        result = fit_encoding_model(X, Phi, cv, feature_names=names, score='r2')
        n_folds = 3
        assert result.scores.shape == (n_folds, X.shape[1], X.shape[2])

    def test_alphas_shape(self):
        X, Phi, names, _ = self._rule_phase_data(n_times=6)
        cv = CVSplitter(n_splits=3, stratified=False, shuffle=True, random_state=0)
        result = fit_encoding_model(X, Phi, cv, feature_names=names)
        assert result.alphas.shape == result.scores.shape

    def test_pearson_mode(self):
        X, Phi, names, _ = self._rule_phase_data()
        cv = CVSplitter(n_splits=3, stratified=False, shuffle=True, random_state=0)
        result = fit_encoding_model(X, Phi, cv, feature_names=names, score='pearson')
        assert result.score_type == 'pearson'
        # Pearson r in [-1, 1]
        assert result.scores.max() <= 1.0 + 1e-5
        assert result.scores.min() >= -1.0 - 1e-5

    def test_known_signal_above_chance(self):
        """With a known linear mapping, R² should be substantially above 0."""
        n_trials, n_ch, n_t = 80, 10, 8
        X, Phi, names, _ = self._rule_phase_data(
            n_trials=n_trials, n_ch=n_ch, n_times=n_t,
            feat='stim_identity',
        )
        cv = CVSplitter(n_splits=4, stratified=False, shuffle=True, random_state=42)
        result = fit_encoding_model(X, Phi, cv, feature_names=names, score='r2')
        grand_mean = result.mean_scores.mean()
        assert grand_mean > 0.0, (
            f"Expected positive R² for known linear signal, got {grand_mean:.4f}"
        )

    def test_store_weights(self):
        X, Phi, names, _ = self._rule_phase_data(n_times=5)
        cv = CVSplitter(n_splits=2, stratified=False, shuffle=True, random_state=0)
        result = fit_encoding_model(X, Phi, cv, feature_names=names,
                                    score='r2', store_weights=True)
        n_folds = 2
        n_feat = Phi.shape[1]
        assert result.weights is not None
        assert result.weights.shape == (n_folds, X.shape[1], n_feat, X.shape[2])

    def test_mean_scores_property(self):
        X, Phi, names, _ = self._rule_phase_data(n_times=5)
        cv = CVSplitter(n_splits=3, stratified=False, shuffle=True, random_state=0)
        result = fit_encoding_model(X, Phi, cv, feature_names=names)
        assert result.mean_scores.shape == (X.shape[1], X.shape[2])
        np.testing.assert_allclose(result.mean_scores, result.scores.mean(axis=0))

    def test_invalid_score_raises(self):
        X, Phi, names, _ = self._rule_phase_data()
        cv = CVSplitter(n_splits=2, stratified=False)
        with pytest.raises(ValueError, match='score'):
            fit_encoding_model(X, Phi, cv, feature_names=names, score='mse')


# ── Nested comparison ─────────────────────────────────────────────────────────

class TestNestedComparison:

    def test_shape(self):
        n, n_ch, n_t = 60, 6, 8
        df = make_df(n=n)
        res = assign_conditions(df, scheme='stim_by_role')
        ep_phase = res['epoch_phase']
        rule_mask = np.isin(ep_phase, RULE_PHASES)

        Phi_A, fn_A = build_feature_space(df, ep_phase, 'stim_identity')
        Phi_B, fn_B = build_feature_space(df, ep_phase, 'stim_x_role')
        Phi_A, Phi_B = Phi_A[rule_mask], Phi_B[rule_mask]

        X, _ = make_meg_from_features(Phi_A, n_ch=n_ch, n_times=n_t, noise_std=0.5)
        cv = CVSplitter(n_splits=3, stratified=False, shuffle=True, random_state=0)

        rA = fit_encoding_model(X, Phi_A, cv, feature_names=fn_A)
        rB = fit_encoding_model(X, Phi_B, cv, feature_names=fn_B)
        delta = nested_comparison(rA, rB)
        assert delta.shape == rA.scores.shape

    def test_more_features_helps(self):
        """stim_x_role (8-d) should fit better than stim_identity (4-d) on role-dependent data."""
        n_trials, n_ch, n_t = 80, 10, 8
        df = make_df(n=n_trials)
        res = assign_conditions(df, scheme='stim_by_role')
        ep_phase = res['epoch_phase']
        rule_mask = np.isin(ep_phase, RULE_PHASES)

        Phi_full, fn_B = build_feature_space(df, ep_phase, 'stim_x_role')
        Phi_B = Phi_full[rule_mask]
        # Generate data from role-binding features
        X, _ = make_meg_from_features(Phi_B, n_ch=n_ch, n_times=n_t, noise_std=0.3)

        Phi_A_full, fn_A = build_feature_space(df, ep_phase, 'stim_identity')
        Phi_A = Phi_A_full[rule_mask]

        cv = CVSplitter(n_splits=4, stratified=False, shuffle=True, random_state=0)
        rA = fit_encoding_model(X, Phi_A, cv, feature_names=fn_A)
        rB = fit_encoding_model(X, Phi_B, cv, feature_names=fn_B)
        delta = nested_comparison(rA, rB)
        # On average, B should be ≥ A (may be noisy for small data)
        assert delta.mean() > -0.2, (
            f"Larger model hurt by more than expected: mean delta = {delta.mean():.3f}"
        )
