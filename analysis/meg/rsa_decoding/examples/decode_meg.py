#!/usr/bin/env python3
"""
decode_meg.py — Sensor-neighborhood MEG decoding for the binding task.

Runs 4 model configs (raw/TF × 3/12 neighbors) across all 8 within-trial
stages for a chosen experiment. Results and plots are written to results/.

Usage
-----
    python decode_meg.py --experiment rule
    python decode_meg.py --experiment A_stim
    python decode_meg.py --experiment response
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

mne.set_log_level('WARNING')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from toolkit import (
    RawAmplitudeFeature,
    TimeFrequencyFeature,
    RidgeLogisticDecoder,
    CVSplitter,
    cross_validate,
)

# ── Paths (edit these) ────────────────────────────────────────────────────────
EPOCHS_PATH = '/jukebox/PNI-classes/students/NEU502/2026-NEU502B/binding/new_ICA/bids/derivatives/analysis4__4/sub-001/ses-01/meg/sub-001_ses-01_task-binding_epo.fif'
BEHAV_PATH  = '/yl0124/projects/NEU502B/neu502b-2025/binding/2026-04-10/sub-001_events.csv'                     # TODO

# ── Constants ─────────────────────────────────────────────────────────────────
STAGE_NAMES = [
    'rule1', 'rule2', 'rule3', 'transition',
    'test1', 'test2', 'test3', 'response',
]
STAGE_OFFSETS = {
    'fixation': 0, 'rule1': 1, 'rule2': 2,  'rule3': 3,
    'transition': 4, 'test1': 5, 'test2': 6, 'test3': 7, 'response': 8,
}
STIM_CATEGORIES = ['circle', 'rectangle', 'star', 'triangle']  # fixed alphabetical order → 0-3

# (feature_type, n_feat_channels, n_skip_channels)
# n_skip keeps sensor groups non-overlapping: skip a wider neighbourhood
# so successive groups are spatially independent.
CONFIGS = [
    ('raw',  3,  24),
    ('raw', 12,  48),
    ('tf',   3,  24),
    ('tf',  12,  48),
]


# ── Labels ────────────────────────────────────────────────────────────────────
def make_label(df: pd.DataFrame, experiment: str):
    """Return (y, metric, n_classes)."""
    if experiment == 'rule':
        y = (df['rule_type'] == 'ABB').values.astype(int)
        return y, 'roc_auc', 2
    elif experiment in ('A_stim', 'B_stim', 'X_stim', 'Y_stim'):
        cat_map = {c: i for i, c in enumerate(STIM_CATEGORIES)}
        y = df[experiment].map(cat_map).values.astype(int)
        return y, 'roc_auc', 4
    elif experiment == 'response':
        y = (df['response_key'] == 2).values.astype(int)
        return y, 'roc_auc', 2
    else:
        raise ValueError(f'Unknown experiment: {experiment!r}')


# ── Sensor utilities ──────────────────────────────────────────────────────────
def top_neighbors(channel_index: int, n: int, pos: np.ndarray) -> np.ndarray:
    """Indices of the n spatially nearest channels (self included)."""
    dists = np.linalg.norm(pos - pos[channel_index], axis=1)
    return np.argsort(dists)[:n]


# ── Feature factories ─────────────────────────────────────────────────────────
def make_raw_feature() -> RawAmplitudeFeature:
    return RawAmplitudeFeature(time_window=(-0.05, 0.4), standardize=True)


def make_tf_feature(n_times: int, sfreq: float) -> TimeFrequencyFeature:
    freqs    = np.arange(4, 80, 4, dtype=float)
    n_cycles = 0.9 * freqs * np.pi * n_times / (5.0 * sfreq)
    return TimeFrequencyFeature(
        freqs=freqs,
        method='morlet',
        n_cycles=n_cycles,
        mode='stacked',
        output='log_power',
        time_window=(0.0, 0.4),
        decimate=2,
        standardize=True,
    )


# ── Core decoding loop ────────────────────────────────────────────────────────
def run_config(
    feature_type: str,
    n_neighbors: int,
    n_skip: int,
    stage_data: list,
    y: np.ndarray,
    metric: str,
    pos3d: np.ndarray,
    sfreq: float,
    times: np.ndarray,
) -> np.ndarray:
    """
    Decode across non-overlapping sensor neighborhoods.

    Returns
    -------
    results : ndarray, shape (n_groups, n_stages, n_times_feature)
    """
    n_channels = pos3d.shape[0]
    n_times_ep = stage_data[0][1].shape[2]

    feature_proto = (
        make_raw_feature() if feature_type == 'raw'
        else make_tf_feature(n_times_ep, sfreq)
    )
    cv              = CVSplitter(n_splits=5, stratified=True, random_state=47)
    decoder_factory = lambda: RidgeLogisticDecoder(C=0.0001)

    channels_done = set()
    results       = []

    for sensor in range(n_channels):
        if sensor in channels_done:
            continue

        feat_ch = top_neighbors(sensor, n_neighbors, pos3d)
        skip_ch = top_neighbors(sensor, n_skip,      pos3d)
        channels_done.update(skip_ch.tolist())

        group_scores = []
        for _stage_name, X_full in stage_data:
            X      = X_full[:, feat_ch]          # (n_trials, n_ch, n_times)
            result = cross_validate(
                feature_proto, decoder_factory,
                X, y, sfreq=sfreq, times=times,
                cv=cv, metric=metric,
            )
            group_scores.append(result.mean_scores)   # (n_times_feature,)

        results.append(np.stack(group_scores, axis=0))  # (n_stages, n_times_feature)

        n_done = len(results)
        if n_done % 10 == 0:
            grand = np.stack(results).mean()
            print(
                f'  [{feature_type}, n={n_neighbors}] {n_done} groups | '
                f'grand mean = {grand:.4f}',
                flush=True,
            )

    return np.stack(results, axis=0)   # (n_groups, n_stages, n_times_feature)


# ── Plotting utilities ────────────────────────────────────────────────────────
def _t_axis(feature_proto, sfreq: float, n_t: int) -> np.ndarray:
    """Reconstruct the time axis from feature parameters (no fit needed)."""
    if isinstance(feature_proto, RawAmplitudeFeature):
        tmin     = feature_proto.time_window[0] if feature_proto.time_window else -0.05
        decimate = feature_proto.decimate
    else:
        tmin     = feature_proto.time_window[0] if feature_proto.time_window else 0.0
        decimate = feature_proto.decimate
    return np.arange(n_t) * decimate / sfreq + tmin


def plot_trace(
    results: np.ndarray,
    experiment: str,
    feature_type: str,
    n_neighbors: int,
    feature_proto,
    sfreq: float,
    out_dir: Path,
) -> None:
    """Concatenated-stage trace plot: mean ± SEM across sensor groups."""
    n_groups, n_stages, n_t = results.shape
    mean = results.mean(axis=0)                     # (n_stages, n_t)
    sem  = results.std(axis=0) / np.sqrt(n_groups)

    t_base    = _t_axis(feature_proto, sfreq, n_t)
    stage_dur = t_base[-1] - t_base[0] + np.diff(t_base[:2])[0]   # one stage width in s
    colors    = plt.cm.tab10(np.linspace(0, 1, n_stages))

    fig, ax = plt.subplots(figsize=(14, 4))

    for k, stage in enumerate(STAGE_NAMES):
        t = t_base + stage_dur * k
        ax.plot(t, mean[k], color=colors[k], label=stage, linewidth=1.2)
        ax.fill_between(t, mean[k] - sem[k], mean[k] + sem[k],
                        alpha=0.2, color=colors[k])
        if k > 0:
            ax.axvline(t[0], color='gray', linewidth=0.5, linestyle=':')

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.9, label='chance')
    ax.set_xlabel('Time within stage (s)  [stages offset for visibility]')
    ax.set_ylabel('ROC-AUC')
    ax.set_title(
        f'Decode: {experiment}  |  {feature_type}  |  {n_neighbors}-ch neighborhoods  '
        f'(n_groups={n_groups})'
    )
    ax.legend(ncol=5, fontsize=8)
    plt.tight_layout()

    path = out_dir / f'trace_{experiment}_{feature_type}_n{n_neighbors}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  → trace plot : {path}')


def plot_boxplot(
    results: np.ndarray,
    experiment: str,
    feature_type: str,
    n_neighbors: int,
    out_dir: Path,
) -> None:
    """One box per stage: distribution over sensor groups of time-averaged ROC-AUC."""
    stage_means = results.mean(axis=2)   # (n_groups, n_stages) — average over time
    colors      = plt.cm.tab10(np.linspace(0, 1, len(STAGE_NAMES)))

    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot(
        [stage_means[:, k] for k in range(len(STAGE_NAMES))],
        labels=STAGE_NAMES,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor((*color[:3], 0.4))

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.9, label='chance')
    ax.set_ylabel('ROC-AUC (time-averaged)')
    ax.set_title(
        f'Decode: {experiment}  |  {feature_type}  |  {n_neighbors}-ch neighborhoods  '
        f'(n_groups={results.shape[0]})'
    )
    ax.legend(fontsize=9)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    path = out_dir / f'boxplot_{experiment}_{feature_type}_n{n_neighbors}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  → boxplot    : {path}')


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MEG neighborhood decoding — binding task')
    parser.add_argument(
        '--experiment', required=True,
        choices=['rule', 'A_stim', 'B_stim', 'X_stim', 'Y_stim', 'response'],
        help='Variable to decode',
    )
    args       = parser.parse_args()
    experiment = args.experiment

    results_dir = Path(__file__).resolve().parent / 'results'
    plot_dir    = results_dir / 'plots'
    results_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f'Loading epochs from {EPOCHS_PATH} …', flush=True)
    epochs   = mne.read_epochs(EPOCHS_PATH, preload=True)
    df       = pd.read_csv(BEHAV_PATH).sort_values('trial').reset_index(drop=True)
    sfreq    = epochs.info['sfreq']
    times    = epochs.times
    n_trials = len(df)

    y, metric, n_classes = make_label(df, experiment)
    print(f'Experiment : {experiment}')
    print(f'Metric     : {metric}  |  n_classes = {n_classes}')
    print(f'Labels     : {dict(zip(*np.unique(y, return_counts=True)))}')

    # ── Sensor positions ──────────────────────────────────────────────────────
    meg_picks = mne.pick_types(epochs.info, meg=True, exclude=[])
    meg_info  = mne.pick_info(epochs.info, meg_picks)
    pos3d     = np.array([ch['loc'][:3] for ch in meg_info['chs']])

    # ── Pre-load all stages ───────────────────────────────────────────────────
    def get_stage(name: str) -> np.ndarray:
        offset = STAGE_OFFSETS[name]
        idx    = [9 * t + offset for t in range(n_trials)]
        return epochs[idx].get_data(picks='meg').astype(np.float32)

    print('Pre-loading stage data …', flush=True)
    stage_data = [(name, get_stage(name)) for name in STAGE_NAMES]
    n_times_ep = stage_data[0][1].shape[2]
    print(f'Stage shape: {stage_data[0][1].shape}  (trials × channels × times)')

    # ── Run all 4 configs ─────────────────────────────────────────────────────
    for feature_type, n_neighbors, n_skip in CONFIGS:
        tag      = f'{experiment}_{feature_type}_n{n_neighbors}'
        out_path = results_dir / f'decode_{tag}.npy'

        if out_path.exists():
            print(f'\n[skip] {tag} — already exists', flush=True)
            continue

        print(f'\n=== {tag} ===', flush=True)
        results = run_config(
            feature_type, n_neighbors, n_skip,
            stage_data, y, metric, pos3d, sfreq, times,
        )

        np.save(out_path, results)
        print(f'Saved  {out_path}  shape={results.shape}')

        feature_proto = (
            make_raw_feature() if feature_type == 'raw'
            else make_tf_feature(n_times_ep, sfreq)
        )
        plot_trace(  results, experiment, feature_type, n_neighbors, feature_proto, sfreq, plot_dir)
        plot_boxplot(results, experiment, feature_type, n_neighbors, plot_dir)

    print('\nAll done.')


if __name__ == '__main__':
    main()
