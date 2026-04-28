"""
Condition coding for perceptXbind MEG RSA and encoding model analyses.

A 'condition' is a discrete combination of task variables that recurs across
multiple trials. The condition coding converts the 120-row behavioral CSV into
per-epoch labels across all 1080 epochs.

Schemes
-------
stim_by_role   : (shape × role) for rule-phase epochs — 8 conditions
phase_by_rule  : (phase × rule_type) for all task-relevant phases — 12 conditions
stim_identity  : shape only, across rule-phase epochs — 4 conditions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

PHASE_NAMES = [
    'fixation', 'rule1', 'rule2', 'rule3', 'transition',
    'test1', 'test2', 'test3', 'response',
]
PHASE_OFFSETS: Dict[str, int] = {name: i for i, name in enumerate(PHASE_NAMES)}

RULE_PHASES = ['rule1', 'rule2', 'rule3']
TEST_PHASES = ['test1', 'test2', 'test3']
STIM_CATEGORIES = ['circle', 'rectangle', 'star', 'triangle']

VALID_SCHEMES = ('stim_by_role', 'phase_by_rule', 'stim_identity', 'stim_by_rule', 'AxBxrule', 'XxYxrule')


# ── Internal helpers ─────────────────────────────────────────────────────────

def _epoch_phases(n_trials: int) -> np.ndarray:
    """Return (n_trials * 9,) object array of phase name per epoch."""
    phases = np.empty(n_trials * 9, dtype=object)
    for t in range(n_trials):
        for offset, name in enumerate(PHASE_NAMES):
            phases[9 * t + offset] = name
    return phases


def _rule_phase_stim_role(df: pd.DataFrame, phase: str):
    """
    Return (stim_array, role_array) for each trial for a rule-phase epoch.

    stim_array : (n_trials,) shape name per trial
    role_array : (n_trials,) 'A' or 'B'
    """
    n = len(df)
    if phase == 'rule1':
        return df['A_stim'].values.copy(), np.full(n, 'A')
    elif phase == 'rule2':
        return df['B_stim'].values.copy(), np.full(n, 'B')
    elif phase == 'rule3':
        stim = np.where(df['rule_type'] == 'ABA', df['A_stim'], df['B_stim'])
        role = np.where(df['rule_type'] == 'ABA', 'A', 'B')
        return stim, role
    else:
        raise ValueError(f"Not a rule phase: {phase!r}")


def _test_phase_stim_role(df: pd.DataFrame, phase: str):
    """Return (stim_array, role_array) for test-phase epochs."""
    n = len(df)
    if phase == 'test1':
        return df['X_stim'].values.copy(), np.full(n, 'X')
    elif phase == 'test2':
        return df['Y_stim'].values.copy(), np.full(n, 'Y')
    elif phase == 'test3':
        # test3 is either X' or Y' depending on test_type (ABA → X-type, ABB → Y-type)
        stim = np.where(df['test_type'] == 'ABA', df['X_stim'], df['Y_stim'])
        role = np.where(df['test_type'] == 'ABA', "X'", "Y'")
        return stim, role
    else:
        raise ValueError(f"Not a test phase: {phase!r}")


# ── Main function ─────────────────────────────────────────────────────────────

def assign_conditions(
    df: pd.DataFrame,
    scheme: str = 'stim_by_role',
    stage_name: Optional[str] = None,
    rule_type: Optional[str] = None,
) -> Dict:
    """
    Assign condition labels to all 1080 epochs.

    Parameters
    ----------
    df     : behavioral DataFrame, 120 rows (one per trial), sorted by trial
    scheme : 'stim_by_role' | 'phase_by_rule' | 'stim_identity'

    Returns
    -------
    dict with keys:
        condition_id       : (1080,) int, −1 for epochs outside the scheme
        condition_names    : list of str indexed by condition id
        epoch_phase        : (1080,) str, within-trial phase for each epoch
        condition_metadata : DataFrame, one row per unique condition
    """
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"scheme must be one of {VALID_SCHEMES}, got {scheme!r}")

    n_trials = len(df)
    n_epochs = n_trials * 9
    epoch_phase = _epoch_phases(n_trials)

    if scheme == 'stim_by_role':
        return _assign_stim_by_role(df, epoch_phase)
    elif scheme == 'phase_by_rule':
        return _assign_phase_by_rule(df, epoch_phase)
    elif scheme == 'stim_by_rule':
        return _assign_stim_by_rule(df, stage_name)
    elif scheme == 'stim_identity':
        return _assign_stim_identity(df, epoch_phase)
    elif scheme == 'AxBxrule':
        return _assign_AxBxrule(df)
    elif scheme == 'XxYxrule':
        return _assign_XxYxrule(df, rule = rule_type)
        


# ── stim_by_role ──────────────────────────────────────────────────────────────

def _assign_stim_by_role(df, epoch_phase):
    """Conditions = {shape} × {role A/B} from rule-phase epochs."""
    n_epochs = len(epoch_phase)
    n_trials = len(df)

    # Build sorted unique (shape, role) pairs appearing in the data
    pairs = set()
    for phase in RULE_PHASES:
        stim, role = _rule_phase_stim_role(df, phase)
        for s, r in zip(stim, role):
            pairs.add((s, r))

    # Sort for determinism: by role then shape alphabetically
    cond_list = sorted(pairs, key=lambda x: (x[1], x[0]))
    cond_map = {pair: i for i, pair in enumerate(cond_list)}
    cond_names = [f"{shape}_{role}" for shape, role in cond_list]

    condition_id = np.full(n_epochs, -1, dtype=int)

    for phase in RULE_PHASES:
        offset = PHASE_OFFSETS[phase]
        stim, role = _rule_phase_stim_role(df, phase)
        for t in range(n_trials):
            pair = (stim[t], role[t])
            condition_id[9 * t + offset] = cond_map[pair]

    meta = pd.DataFrame(
        [(i, s, r) for i, (s, r) in enumerate(cond_list)],
        columns=['cond_id', 'shape', 'role'],
    )

    return {
        'condition_id':       condition_id,
        'condition_names':    cond_names,
        'epoch_phase':        epoch_phase,
        'condition_metadata': meta,
    }


def _assign_stim_by_rule(df, stage_name):
    """Conditions = {shape at stage_name} × {rule ABA/ABB} — 8 conditions."""
    if stage_name not in ['A', 'B', 'X', 'Y']:
        raise ValueError(f"stage_name must be one of 'A', 'B', 'X', 'Y', got {stage_name!r}")

    n_trials = len(df)
    stim = df[f"{stage_name}_stim"].values.copy()
    rule_types = ['ABA', 'ABB']

    cond_list = [(shape, rule) for shape in STIM_CATEGORIES for rule in rule_types]
    cond_map = {pair: i for i, pair in enumerate(cond_list)}
    cond_names = [f"{shape}_{rule}" for shape, rule in cond_list]

    condition_id = np.full(n_trials, -1, dtype=int)
    for t, (shape, rule) in enumerate(zip(stim, df['rule_type'].values)):
        condition_id[t] = cond_map[(shape, rule)]

    meta = pd.DataFrame(
        [(i, shape, rule, stage_name) for i, (shape, rule) in enumerate(cond_list)],
        columns=['cond_id', 'shape', 'rule_type', 'stage_name'],
    )

    return {
        'condition_id':       condition_id,
        'condition_names':    cond_names,
        'epoch_phase':        np.full(n_trials, stage_name, dtype=object),
        'condition_metadata': meta,
    }


def _assign_AxBxrule(df):
    """Conditions = {A shape} × {B shape} × {rule ABA/ABB} — 24 conditions."""
    n_trials = len(df)
    rule_types = ['ABA', 'ABB']

    cond_list = [
        (a_shape, b_shape, rule)
        for a_shape in STIM_CATEGORIES
        for b_shape in STIM_CATEGORIES
        for rule in rule_types
    ]
    print(len(cond_list))
    cond_map = {triple: i for i, triple in enumerate(cond_list)}
    cond_names = [f"{a}_{b}_{rule}" for a, b, rule in cond_list]

    condition_id = np.full(n_trials, -1, dtype=int)
    for t, (a_shape, b_shape, rule) in enumerate(
        zip(df['A_stim'].values, df['B_stim'].values, df['rule_type'].values)
    ):
        condition_id[t] = cond_map[(a_shape, b_shape, rule)]

    meta = pd.DataFrame(
        [(i, a, b, rule) for i, (a, b, rule) in enumerate(cond_list)],
        columns=['cond_id', 'A_shape', 'B_shape', 'rule_type'],
    )

    return {
        'condition_id':       condition_id,
        'condition_names':    cond_names,
        'epoch_phase':        np.full(n_trials, 'AxBxrule', dtype=object),
        'condition_metadata': meta,
    }

def _assign_XxYxrule(df, rule = "rtmatch"):
    """Conditions = {X shape} × {Y shape} × {rule in {rtmatch, YYmatch, orderrandom, orderedrandom}} — 24 conditions."""
    n_trials = len(df)
    if rule == "rtmatch":
        rule_types = ['match', 'mismatch']
        rule_values = np.where(df['match'], 'match', 'mismatch')
    elif rule == "YYmatch":
        rule_types = ['XYY', 'XY*']
        rule_values = np.where(np.logical_or(np.logical_and(df['rule_type'] == 'ABB', df['match'] == 1), np.logical_and(df['rule_type'] == 'ABA', df['match'] == 0, df['mismatch_type'] == 'rule_order')), 'XYY', 'XY*')
    elif rule == "orderrandom":
        rule_types = ['XY*', "XYZ", "match"]
        rule_values = np.where(df['mismatch_type'] == 'random', 'XYZ', np.where(df['match'] == 0, 'XY*', 'match'))
    elif rule == "orderedrandom":
        rule_types = ['XYX', 'XYY', 'XYZ']
        rule_values = np.where(df['mismatch_type'] == 'random', 'XYZ', np.where(np.logical_or(np.logical_and(df['rule_type'] == 'ABB', df['match'] == 1), np.logical_and(df['rule_type'] == 'ABA', df['match'] == 0, df['mismatch_type'] == 'rule_order')), 'XYY', 'XYX'))

    cond_list = [
        (x_shape, y_shape, rule)
        for x_shape in STIM_CATEGORIES
        for y_shape in STIM_CATEGORIES
        for rule in rule_types
    ]
    print(len(cond_list))
    cond_map = {triple: i for i, triple in enumerate(cond_list)}
    cond_names = [f"{x}_{y}_{rule}" for x, y, rule in cond_list]

    condition_id = np.full(n_trials, -1, dtype=int)

    for t, (x_shape, y_shape, rule) in enumerate(
        zip(df['X_stim'].values, df['Y_stim'].values, rule_values)
    ):
        condition_id[t] = cond_map[(x_shape, y_shape, rule)]

    meta = pd.DataFrame(
        [(i, x, y, rule) for i, (x, y, rule) in enumerate(cond_list)],
        columns=['cond_id', 'X_shape', 'Y_shape', 'rule_type'],
    )

    return {
        'condition_id':       condition_id,
        'condition_names':    cond_names,
        'epoch_phase':        np.full(n_trials, 'XxYxrule', dtype=object),
        'condition_metadata': meta,
    }


# def _assign_stim_by_rule_by_response(df, stage_name):


# ── phase_by_rule ─────────────────────────────────────────────────────────────

def _assign_phase_by_rule(df, epoch_phase):
    """Conditions = {rule1..test3} × {ABA, ABB} — 12 conditions."""
    n_epochs = len(epoch_phase)
    n_trials = len(df)

    task_phases = RULE_PHASES + TEST_PHASES
    rule_types  = sorted(df['rule_type'].unique())  # ['ABA', 'ABB']

    cond_list  = [(ph, rt) for ph in task_phases for rt in rule_types]
    cond_map   = {pair: i for i, pair in enumerate(cond_list)}
    cond_names = [f"{ph}_{rt}" for ph, rt in cond_list]

    condition_id = np.full(n_epochs, -1, dtype=int)

    for phase in task_phases:
        offset = PHASE_OFFSETS[phase]
        for t, rt in enumerate(df['rule_type'].values):
            condition_id[9 * t + offset] = cond_map[(phase, rt)]

    meta = pd.DataFrame(
        [(i, ph, rt) for i, (ph, rt) in enumerate(cond_list)],
        columns=['cond_id', 'phase', 'rule_type'],
    )

    return {
        'condition_id':       condition_id,
        'condition_names':    cond_names,
        'epoch_phase':        epoch_phase,
        'condition_metadata': meta,
    }


# ── stim_identity ─────────────────────────────────────────────────────────────

def _assign_stim_identity(df, epoch_phase):
    """Conditions = {circle, rectangle, star, triangle} — rule-phase epochs only."""
    n_epochs = len(epoch_phase)
    n_trials = len(df)

    cond_list = sorted(STIM_CATEGORIES)
    cond_map  = {s: i for i, s in enumerate(cond_list)}
    cond_names = cond_list[:]

    condition_id = np.full(n_epochs, -1, dtype=int)

    for phase in RULE_PHASES:
        offset = PHASE_OFFSETS[phase]
        stim, _ = _rule_phase_stim_role(df, phase)
        for t in range(n_trials):
            condition_id[9 * t + offset] = cond_map[stim[t]]

    meta = pd.DataFrame(
        [(i, s) for i, s in enumerate(cond_list)],
        columns=['cond_id', 'shape'],
    )

    return {
        'condition_id':       condition_id,
        'condition_names':    cond_names,
        'epoch_phase':        epoch_phase,
        'condition_metadata': meta,
    }


# ── Condition count helper ────────────────────────────────────────────────────

def condition_counts(
    condition_result: Dict,
    phase_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of trial counts per condition (per phase if filtered).

    Parameters
    ----------
    condition_result : output of assign_conditions
    phase_filter     : list of phase names to include; None = all

    Returns
    -------
    DataFrame with columns [cond_id, condition_name, phase, count]
    """
    cids    = condition_result['condition_id']
    phases  = condition_result['epoch_phase']
    names   = condition_result['condition_names']

    rows = []
    mask = np.ones(len(cids), dtype=bool)
    if phase_filter is not None:
        mask = np.isin(phases, phase_filter)

    unique_phases = sorted(set(phases[mask & (cids >= 0)]))
    for ph in unique_phases:
        ph_mask = (phases == ph) & (cids >= 0)
        for cid in np.unique(cids[ph_mask]):
            cnt = ((phases == ph) & (cids == cid)).sum()
            rows.append({'cond_id': cid, 'condition_name': names[cid],
                         'phase': ph, 'count': cnt})

    return pd.DataFrame(rows)
