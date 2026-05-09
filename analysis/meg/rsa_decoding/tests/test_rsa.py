"""
Tests for toolkit/rsa.py.

Uses synthetic data with known RDM structure so correctness is verifiable.
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from toolkit.rsa import (
    RSAResult,
    build_model_rdm,
    check_model_rdm_collinearity,
    time_resolved_rsa,
    _correlation_rdm_series,
    _crossnobis_rdm_series,
)
from toolkit.cv import CVSplitter
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_metadata_stim_role(shapes=None, roles=None):
    if shapes is None:
        shapes = ['circle', 'rectangle', 'star', 'triangle']
    if roles is None:
        roles = ['A', 'B']
    rows = [(i, s, r)
            for i, (s, r) in enumerate(
                (s, r) for r in roles for s in shapes
            )]
    return pd.DataFrame(rows, columns=['cond_id', 'shape', 'role'])


def make_synthetic_meg(n_per_cond=20, n_ch=10, n_times=15, n_conds=4, seed=0):
    """
    Synthetic MEG where each condition has a distinct mean pattern.
    condition_ids are 0..n_conds-1, balanced.
    """
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((n_conds, n_ch)) * 3.0
    X_list, y_list = [], []
    for c in range(n_conds):
        noise = rng.standard_normal((n_per_cond, n_ch, n_times)).astype(np.float32)
        signal = means[c, :, np.newaxis]  # broadcast over time
        X_list.append(noise + signal)
        y_list.append(np.full(n_per_cond, c, dtype=int))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    return X, y, means


# ── Model RDM construction ────────────────────────────────────────────────────

class TestBuildModelRdm:

    def test_stimulus_identity_shape(self):
        meta = make_metadata_stim_role()
        rdm = build_model_rdm(meta, 'stimulus_identity')
        K = len(meta)
        assert rdm.shape == (K, K)

    def test_stimulus_identity_zero_diagonal(self):
        meta = make_metadata_stim_role()
        rdm = build_model_rdm(meta, 'stimulus_identity')
        np.testing.assert_array_equal(np.diag(rdm), 0)

    def test_stimulus_identity_symmetric(self):
        meta = make_metadata_stim_role()
        rdm = build_model_rdm(meta, 'stimulus_identity')
        np.testing.assert_array_equal(rdm, rdm.T)

    def test_stimulus_identity_values(self):
        meta = make_metadata_stim_role(shapes=['circle', 'star'], roles=['A'])
        rdm = build_model_rdm(meta, 'stimulus_identity')
        # cond 0: circle_A, cond 1: star_A → different shapes → d=1
        assert rdm[0, 1] == 1.0
        assert rdm[0, 0] == 0.0

    def test_abstract_role_same_role_zero(self):
        meta = make_metadata_stim_role(shapes=['circle', 'star'], roles=['A', 'B'])
        rdm = build_model_rdm(meta, 'abstract_role')
        # A/A pairs should be 0
        cids_A = meta[meta['role'] == 'A']['cond_id'].values
        for i in cids_A:
            for j in cids_A:
                assert rdm[i, j] == 0.0

    def test_abstract_role_different_role_one(self):
        meta = make_metadata_stim_role(shapes=['circle'], roles=['A', 'B'])
        rdm = build_model_rdm(meta, 'abstract_role')
        assert rdm[0, 1] == 1.0

    def test_conjunctive_binding_matches_identity(self):
        # With (shape × role) conditions, conjunctive = identity matrix
        meta = make_metadata_stim_role(shapes=['circle', 'star'], roles=['A', 'B'])
        rdm = build_model_rdm(meta, 'conjunctive_binding')
        K = len(meta)
        for i in range(K):
            for j in range(K):
                same = (meta['shape'].iloc[i] == meta['shape'].iloc[j] and
                        meta['role'].iloc[i] == meta['role'].iloc[j])
                expected = 0.0 if same else 1.0
                assert rdm[i, j] == expected

    def test_unknown_model_raises(self):
        meta = make_metadata_stim_role()
        with pytest.raises(ValueError):
            build_model_rdm(meta, 'nonsense_model')

    def test_missing_column_raises(self):
        meta = pd.DataFrame({'cond_id': [0, 1], 'shape': ['circle', 'star']})
        with pytest.raises(ValueError, match='role'):
            build_model_rdm(meta, 'abstract_role')


# ── VIF check ─────────────────────────────────────────────────────────────────

class TestVIF:

    def test_independent_models_low_vif(self):
        meta = make_metadata_stim_role()
        rdms = {
            'stimulus_identity': build_model_rdm(meta, 'stimulus_identity'),
            'abstract_role':     build_model_rdm(meta, 'abstract_role'),
        }
        df_vif = check_model_rdm_collinearity(rdms)
        assert set(df_vif.columns) == {'model', 'vif'}
        assert len(df_vif) == 2

    def test_identical_models_high_vif(self):
        meta = make_metadata_stim_role()
        rdm = build_model_rdm(meta, 'stimulus_identity')
        rdms = {'model_A': rdm.copy(), 'model_B': rdm.copy()}
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            df_vif = check_model_rdm_collinearity(rdms)
            high = df_vif[df_vif['vif'] > 5]
            assert len(high) > 0 or len(w) > 0  # either high VIF or warning


# ── Neural RDM computation ────────────────────────────────────────────────────

class TestCorrelationRDM:

    def test_shape(self):
        X, y, _ = make_synthetic_meg(n_per_cond=10, n_ch=8, n_times=12, n_conds=4)
        classes = np.unique(y)
        rdm = _correlation_rdm_series(X, y, classes)
        assert rdm.shape == (12, 4, 4)

    def test_zero_diagonal(self):
        X, y, _ = make_synthetic_meg(n_per_cond=10, n_ch=8, n_times=12, n_conds=4)
        classes = np.unique(y)
        rdm = _correlation_rdm_series(X, y, classes)
        for t in range(12):
            np.testing.assert_array_almost_equal(np.diag(rdm[t]), 0)

    def test_symmetric(self):
        X, y, _ = make_synthetic_meg(n_per_cond=10, n_ch=8, n_times=12, n_conds=4)
        classes = np.unique(y)
        rdm = _correlation_rdm_series(X, y, classes)
        for t in range(12):
            np.testing.assert_array_almost_equal(rdm[t], rdm[t].T)

    def test_values_in_range(self):
        X, y, _ = make_synthetic_meg(n_per_cond=20, n_ch=8, n_times=5, n_conds=3)
        classes = np.unique(y)
        rdm = _correlation_rdm_series(X, y, classes)
        assert rdm.min() >= -1.0 - 1e-6
        assert rdm.max() <= 2.0 + 1e-6  # 1 - (-1) = 2


class TestCrossnobisRDM:

    def test_shape(self):
        X, y, _ = make_synthetic_meg(n_per_cond=20, n_ch=8, n_times=10, n_conds=4)
        classes = np.unique(y)
        cv = CVSplitter(n_splits=4, stratified=True, shuffle=True, random_state=0)
        rdm = _crossnobis_rdm_series(X, y, classes, cv)
        assert rdm.shape == (10, 4, 4)

    def test_zero_diagonal(self):
        X, y, _ = make_synthetic_meg(n_per_cond=20, n_ch=8, n_times=10, n_conds=4)
        classes = np.unique(y)
        cv = CVSplitter(n_splits=4, stratified=True, shuffle=True, random_state=0)
        rdm = _crossnobis_rdm_series(X, y, classes, cv)
        for t in range(10):
            np.testing.assert_array_almost_equal(np.diag(rdm[t]), 0)

    def test_symmetric(self):
        X, y, _ = make_synthetic_meg(n_per_cond=20, n_ch=8, n_times=10, n_conds=4)
        classes = np.unique(y)
        cv = CVSplitter(n_splits=4, stratified=True, shuffle=True, random_state=0)
        rdm = _crossnobis_rdm_series(X, y, classes, cv)
        for t in range(10):
            np.testing.assert_array_almost_equal(rdm[t], rdm[t].T)

    def test_null_near_zero(self):
        """Under null (identical conditions), crossnobis should be near 0."""
        rng = np.random.default_rng(42)
        n_per = 40
        # All epochs are pure noise (no condition effect)
        X = rng.standard_normal((n_per * 4, 8, 10)).astype(np.float32)
        y = np.repeat(np.arange(4), n_per)
        classes = np.arange(4)
        cv = CVSplitter(n_splits=4, stratified=True, shuffle=True, random_state=0)
        rdm = _crossnobis_rdm_series(X, y, classes, cv)
        triu = np.triu_indices(4, k=1)
        # Mean distance across upper triangle and time should be close to 0
        mean_d = rdm[:, triu[0], triu[1]].mean()
        assert abs(mean_d) < 2.0, f"Null crossnobis mean {mean_d:.3f} unexpectedly large"

    def test_signal_positive(self):
        """With strong signals, within-class distances should be lower than between-class."""
        X, y, _ = make_synthetic_meg(
            n_per_cond=30, n_ch=15, n_times=5, n_conds=4, seed=7
        )
        classes = np.arange(4)
        cv = CVSplitter(n_splits=4, stratified=True, shuffle=True, random_state=0)
        rdm = _crossnobis_rdm_series(X, y, classes, cv)
        # Average off-diagonal (between-condition) distance > 0 at most timepoints
        triu = np.triu_indices(4, k=1)
        mean_between = rdm[:, triu[0], triu[1]].mean()
        assert mean_between > 0, f"Expected positive between-cond distances, got {mean_between:.3f}"


# ── time_resolved_rsa ─────────────────────────────────────────────────────────

class TestTimeResolvedRSA:

    def _setup(self, n_per_cond=25, n_ch=10, n_times=12, seed=3):
        """4-condition data with known structure: conditions differ in shape identity."""
        rng = np.random.default_rng(seed)
        # Each condition = one shape; make distinct mean patterns
        means = rng.standard_normal((4, n_ch)) * 4.0
        X_parts, y_parts = [], []
        for c in range(4):
            noise = rng.standard_normal((n_per_cond, n_ch, n_times)).astype(np.float32)
            X_parts.append(noise + means[c, :, np.newaxis])
            y_parts.append(np.full(n_per_cond, c))
        X = np.concatenate(X_parts)
        y = np.concatenate(y_parts)
        return X, y

    def test_result_type(self):
        X, y = self._setup()
        meta = make_metadata_stim_role(shapes=['circle','rect','star','tri'], roles=['A'])
        meta = meta[meta['role'] == 'A'].reset_index(drop=True)
        meta['cond_id'] = range(4)
        model_rdms = {'stim': build_model_rdm(meta, 'stimulus_identity')}
        result = time_resolved_rsa(X, y, model_rdms,
                                   times=np.linspace(0, 0.1, 12),
                                   method='spearman', distance='correlation')
        assert isinstance(result, RSAResult)

    def test_fits_shape_spearman(self):
        X, y = self._setup()
        meta = make_metadata_stim_role(shapes=['circle','rect','star','tri'], roles=['A'])
        meta = meta[meta['role'] == 'A'].reset_index(drop=True)
        meta['cond_id'] = range(4)
        model_rdms = {
            'stim':  build_model_rdm(meta, 'stimulus_identity'),
        }
        result = time_resolved_rsa(X, y, model_rdms,
                                   times=np.linspace(0, 0.1, 12),
                                   method='spearman', distance='correlation')
        assert result.fits.shape == (1, 12)

    def test_fits_shape_regression(self):
        X, y = self._setup()
        meta = make_metadata_stim_role(shapes=['circle','rect','star','tri'], roles=['A'])
        meta = meta[meta['role'] == 'A'].reset_index(drop=True)
        meta['cond_id'] = range(4)
        model_rdms = {
            'stim': build_model_rdm(meta, 'stimulus_identity'),
        }
        result = time_resolved_rsa(X, y, model_rdms,
                                   times=np.linspace(0, 0.1, 12),
                                   method='regression', distance='crossnobis',
                                   cv=CVSplitter(n_splits=4, stratified=True))
        assert result.fits.shape == (1, 12)

    def test_store_rdms(self):
        X, y = self._setup()
        meta = make_metadata_stim_role(shapes=['circle','rect','star','tri'], roles=['A'])
        meta = meta[meta['role'] == 'A'].reset_index(drop=True)
        meta['cond_id'] = range(4)
        model_rdms = {'stim': build_model_rdm(meta, 'stimulus_identity')}
        result = time_resolved_rsa(X, y, model_rdms,
                                   times=np.linspace(0, 0.1, 12),
                                   distance='correlation', store_rdms=True)
        assert result.neural_rdms is not None
        assert result.neural_rdms.shape == (12, 4, 4)

    def test_minus1_raises(self):
        X, y = self._setup()
        y_bad = y.copy()
        y_bad[0] = -1
        model_rdms = {'stim': np.ones((4, 4)) - np.eye(4)}
        with pytest.raises(ValueError, match='-1'):
            time_resolved_rsa(X, y_bad, model_rdms, times=np.linspace(0, 0.1, 12))

    def test_known_structure_spearman(self):
        """With strong shape signal, stimulus_identity model should have high ρ.

        Use 8 conditions (4 shapes × 2 roles) so that shape-matched pairs
        (circle_A vs circle_B) appear in the model RDM as distance=0, giving
        a non-constant model vector that can produce non-trivial Spearman ρ.
        """
        n_ch, n_t = 20, 5
        rng = np.random.default_rng(0)
        shapes = ['circle', 'rectangle', 'star', 'triangle']
        # Distinct mean pattern per shape (same across roles → only shape encoded)
        shape_means = rng.standard_normal((4, n_ch)).astype(np.float32) * 5.0
        # 8 conditions: (circle,A), (circle,B), (rect,A), (rect,B), ...
        cond_means = np.concatenate([shape_means, shape_means], axis=0)  # 8 conds

        X_parts, y_parts = [], []
        for c in range(8):
            noise = rng.standard_normal((20, n_ch, n_t)).astype(np.float32) * 0.5
            X_parts.append(noise + cond_means[c, :, np.newaxis])
            y_parts.append(np.full(20, c))
        X = np.concatenate(X_parts)
        y = np.concatenate(y_parts)

        meta = pd.DataFrame({
            'cond_id': list(range(8)),
            'shape':   shapes * 2,          # circle,rect,star,tri,circle,rect,star,tri
            'role':    ['A'] * 4 + ['B'] * 4,
        })
        model_rdms = {'stim': build_model_rdm(meta, 'stimulus_identity')}
        result = time_resolved_rsa(X, y, model_rdms,
                                   times=np.linspace(0, 0.05, n_t),
                                   method='spearman', distance='correlation')
        # Model predicts 0-distance for same-shape pairs (A vs B of same shape)
        # Neural RDM should reflect this (same shape → small distance)
        assert result.fits[0].mean() > 0.3, (
            f"Expected high Spearman ρ on high-SNR shape data, got {result.fits[0].mean():.3f}"
        )
