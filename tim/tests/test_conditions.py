"""
Tests for toolkit/conditions.py.

Verifies condition assignment, metadata structure, and trial counts
against the real behavioral CSV layout.
"""

import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from toolkit.conditions import (
    assign_conditions, condition_counts,
    PHASE_NAMES, PHASE_OFFSETS, RULE_PHASES,
)

# ── Synthetic behavioral DataFrame ───────────────────────────────────────────

def make_df(n=40, seed=0):
    """Minimal synthetic behavioral DataFrame mimicking the real CSV."""
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
        rows.append({
            'trial': t, 'rule_type': rt, 'test_type': tt,
            'A_stim': a, 'B_stim': b, 'X_stim': x, 'Y_stim': y,
        })
    return pd.DataFrame(rows)


# ── epoch_phase ───────────────────────────────────────────────────────────────

class TestEpochPhase:

    def test_total_epochs(self):
        df = make_df(n=10)
        result = assign_conditions(df, scheme='stim_by_role')
        assert len(result['epoch_phase']) == 10 * 9

    def test_phase_cycle(self):
        df = make_df(n=5)
        phases = assign_conditions(df, scheme='stim_by_role')['epoch_phase']
        for t in range(5):
            for offset, name in enumerate(PHASE_NAMES):
                assert phases[9 * t + offset] == name

    def test_all_schemes_same_epoch_phase(self):
        df = make_df(n=20)
        ph1 = assign_conditions(df, scheme='stim_by_role')['epoch_phase']
        ph2 = assign_conditions(df, scheme='phase_by_rule')['epoch_phase']
        ph3 = assign_conditions(df, scheme='stim_identity')['epoch_phase']
        np.testing.assert_array_equal(ph1, ph2)
        np.testing.assert_array_equal(ph1, ph3)


# ── stim_by_role ──────────────────────────────────────────────────────────────

class TestStimByRole:

    def test_n_conditions(self):
        df = make_df(n=40)
        result = assign_conditions(df, scheme='stim_by_role')
        # At most 4 shapes × 2 roles = 8 conditions
        n_conds = len(result['condition_names'])
        assert 2 <= n_conds <= 8

    def test_metadata_columns(self):
        df = make_df(n=40)
        meta = assign_conditions(df, scheme='stim_by_role')['condition_metadata']
        assert set(meta.columns) >= {'cond_id', 'shape', 'role'}

    def test_rule_phases_have_valid_ids(self):
        df = make_df(n=20)
        result = assign_conditions(df, scheme='stim_by_role')
        cids = result['condition_id']
        phases = result['epoch_phase']
        for phase in RULE_PHASES:
            mask = phases == phase
            assert (cids[mask] >= 0).all(), f"Negative cids in {phase}"

    def test_non_rule_phases_are_minus1(self):
        df = make_df(n=20)
        result = assign_conditions(df, scheme='stim_by_role')
        cids = result['condition_id']
        phases = result['epoch_phase']
        non_rule = [p for p in PHASE_NAMES if p not in RULE_PHASES]
        for phase in non_rule:
            mask = phases == phase
            assert (cids[mask] == -1).all(), f"Expected -1 in {phase}"

    def test_condition_names_match_metadata(self):
        df = make_df(n=40)
        result = assign_conditions(df, scheme='stim_by_role')
        names = result['condition_names']
        meta  = result['condition_metadata']
        assert len(names) == len(meta)
        for _, row in meta.iterrows():
            expected = f"{row['shape']}_{row['role']}"
            assert names[row['cond_id']] == expected

    def test_rule1_always_role_A(self):
        df = make_df(n=30)
        result = assign_conditions(df, scheme='stim_by_role')
        cids   = result['condition_id']
        phases = result['epoch_phase']
        meta   = result['condition_metadata']

        rule1_cids = cids[phases == 'rule1']
        for cid in rule1_cids:
            role = meta.loc[meta['cond_id'] == cid, 'role'].values[0]
            assert role == 'A', f"rule1 epoch had role={role!r}, expected A"

    def test_rule2_always_role_B(self):
        df = make_df(n=30)
        result = assign_conditions(df, scheme='stim_by_role')
        cids   = result['condition_id']
        phases = result['epoch_phase']
        meta   = result['condition_metadata']

        rule2_cids = cids[phases == 'rule2']
        for cid in rule2_cids:
            role = meta.loc[meta['cond_id'] == cid, 'role'].values[0]
            assert role == 'B', f"rule2 epoch had role={role!r}, expected B"

    def test_rule3_role_matches_rule_type(self):
        """ABA → rule3 role = A; ABB → rule3 role = B."""
        df = make_df(n=40)
        result = assign_conditions(df, scheme='stim_by_role')
        cids   = result['condition_id']
        phases = result['epoch_phase']
        meta   = result['condition_metadata']

        for t, row in df.iterrows():
            ep_idx = 9 * t + PHASE_OFFSETS['rule3']
            cid = cids[ep_idx]
            role = meta.loc[meta['cond_id'] == cid, 'role'].values[0]
            expected_role = 'A' if row['rule_type'] == 'ABA' else 'B'
            assert role == expected_role, (
                f"Trial {t}: rule_type={row['rule_type']}, expected role={expected_role}, "
                f"got {role}"
            )


# ── phase_by_rule ─────────────────────────────────────────────────────────────

class TestPhaseByRule:

    def test_n_conditions(self):
        df = make_df(n=40)
        result = assign_conditions(df, scheme='phase_by_rule')
        # 6 task phases × 2 rule types = 12
        assert len(result['condition_names']) == 12

    def test_metadata_columns(self):
        df = make_df(n=40)
        meta = assign_conditions(df, scheme='phase_by_rule')['condition_metadata']
        assert set(meta.columns) >= {'cond_id', 'phase', 'rule_type'}

    def test_fixation_and_transition_are_minus1(self):
        df = make_df(n=20)
        result = assign_conditions(df, scheme='phase_by_rule')
        cids = result['condition_id']
        phases = result['epoch_phase']
        for ph in ('fixation', 'transition', 'response'):
            mask = phases == ph
            assert (cids[mask] == -1).all()

    def test_task_phases_all_valid(self):
        df = make_df(n=20)
        result = assign_conditions(df, scheme='phase_by_rule')
        cids   = result['condition_id']
        phases = result['epoch_phase']
        task_phases = ['rule1','rule2','rule3','test1','test2','test3']
        for ph in task_phases:
            mask = phases == ph
            assert (cids[mask] >= 0).all()


# ── stim_identity ─────────────────────────────────────────────────────────────

class TestStimIdentity:

    def test_n_conditions(self):
        df = make_df(n=40)
        result = assign_conditions(df, scheme='stim_identity')
        assert len(result['condition_names']) == 4

    def test_metadata_has_shape(self):
        df = make_df(n=40)
        meta = assign_conditions(df, scheme='stim_identity')['condition_metadata']
        assert 'shape' in meta.columns


# ── condition_counts ──────────────────────────────────────────────────────────

class TestConditionCounts:

    def test_count_structure(self):
        df = make_df(n=40)
        result = assign_conditions(df, scheme='stim_by_role')
        counts = condition_counts(result, phase_filter=['rule1'])
        assert 'count' in counts.columns
        assert 'cond_id' in counts.columns
        assert (counts['count'] > 0).all()

    def test_total_rule_phase_count(self):
        n = 30
        df = make_df(n=n)
        result = assign_conditions(df, scheme='stim_by_role')
        counts = condition_counts(result, phase_filter=RULE_PHASES)
        # Total should equal n_trials * 3 (one epoch per rule phase per trial)
        assert counts['count'].sum() == n * 3

    def test_invalid_scheme(self):
        df = make_df(n=10)
        with pytest.raises(ValueError, match='scheme'):
            assign_conditions(df, scheme='nonsense')
