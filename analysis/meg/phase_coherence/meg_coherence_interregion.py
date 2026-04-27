"""
Inter-Region Spectral Coherence Pipeline
=========================================
Mirrors the phase-locked ITC pipeline but computes pairwise spectral coherence
BETWEEN sensor regions rather than within-region ITC.

For each region pair (occipital-parietal, occipital-frontal, parietal-frontal):
  - Computes mean cross-spectral coherence across all channel pairs spanning the
    two regions, separately for match, rule_order, and random conditions.
  - Plots ITC(Match) - ITC(Rule-Order) and ITC(Match) - ITC(Random) as
    time-frequency maps.
  - Significance via Z-score permutation test (same logic as ITC pipeline):
      Z > 1.96  : match significantly more coherent than non-match
      Z < -1.96 : non-match significantly more coherent than match

Outputs one PNG per region pair:
  coherence_occipital-parietal.png
  coherence_occipital-frontal.png
  coherence_parietal-frontal.png

Dependencies
------------
  mne >= 1.6, numpy, pandas, matplotlib, itertools
"""

import argparse
import re
import itertools
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import mne

mne.set_log_level('WARNING')

# ── USER SETTINGS ──────────────────────────────────────────────────────────────

EPOCHS_FILE = pathlib.Path(
    '/jukebox/PNI-classes/students/NEU502/2026-NEU502B/binding/data/binding/'
    'bids/derivatives/analysis1__1/sub-001/ses-01/meg/'
    'sub-001_ses-01_task-binding_proc-clean_epo.fif'
)
BEH_FILE = pathlib.Path(
    '/usr/people/ao8210/neu502b/brain-binding/data/2026-04-10/sub-001_events.csv'
)

# Region definitions — same as ITC pipeline
REGIONS = {
    'occipital' : [r'^O', r'^Iz'],
    'parietal'  : [r'^P'],
    'frontal'   : [r'^F[^p]', r'^AF'],
}

# Region pairs to analyse — all combinations by default
REGION_PAIRS = list(itertools.combinations(REGIONS.keys(), 2))
# e.g. [('occipital', 'parietal'), ('occipital', 'frontal'), ('parietal', 'frontal')]

# None = all axes (X, Y, Z); ' Z' = radial only
CH_SUFFIX = None

# TFR / coherence parameters
FREQS          = np.arange(8, 41, 1)
N_CYCLES       = FREQS / 4.0
TIME_BANDWIDTH = 2.0
DECIM          = 1

# Set to 0 to skip permutation testing
N_PERMUTATIONS = 100

# Plot limits (seconds)
TMIN_PLOT = -0.100
TMAX_PLOT =  0.400

# Highlight boxes: (t_start_s, t_end_s, f_low_hz, f_high_hz)
ALPHA_BOX = (-0.05, 0.150,  8, 12)
BETA_BOX  = ( 0.00, 0.250, 13, 30)

VLINES = [0.0]

OUTPUT_DIR = pathlib.Path('.')
DPI = 150

# ── END USER SETTINGS ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',    type=pathlib.Path, default=None)
    p.add_argument('--beh',       type=pathlib.Path, default=None)
    p.add_argument('--outputdir', type=pathlib.Path, default=None)
    args, _ = p.parse_known_args()
    return args

_args      = _parse_args()
EPOCHS_FILE = _args.epochs    or EPOCHS_FILE
BEH_FILE    = _args.beh       or BEH_FILE
OUTPUT_DIR  = _args.outputdir or OUTPUT_DIR


# ── CHANNEL SELECTION ──────────────────────────────────────────────────────────

def channels_for_region(all_ch_names, patterns, ch_suffix=None):
    """Return channels matching any regex pattern, optionally filtered by suffix."""
    matched = [ch for ch in all_ch_names
               if any(re.match(pat, ch) for pat in patterns)]
    if ch_suffix is not None:
        matched = [ch for ch in matched if ch.endswith(ch_suffix)]
    return matched


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_and_label(epochs_path, beh_path):
    """Load epochs and split test_non_match into rule_order / random subtypes.
    Identical to ITC pipeline."""
    epo = mne.read_epochs(epochs_path, preload=True, verbose=False)
    df  = pd.read_csv(beh_path).sort_values('trial').reset_index(drop=True)

    meta      = epo.metadata.copy()
    test_mask = meta['event_name'].isin(['test_match', 'test_non_match'])
    meta_test = meta[test_mask]

    epo_seq = (meta_test['event_name'] == 'test_match').astype(int).values
    if not np.array_equal(epo_seq, df['match'].values):
        raise ValueError('Match/non-match sequence does not match behavioural CSV.')

    meta['mismatch_type'] = ''
    meta.loc[meta_test.index, 'mismatch_type'] = df['mismatch_type'].fillna('').values
    epo.metadata = meta

    epo_match = epo['test_match']
    epo_ro    = epo[epo.metadata['mismatch_type'] == 'rule_order']
    epo_rand  = epo[epo.metadata['mismatch_type'] == 'random']

    print(f'Epoch counts - match: {len(epo_match)}, '
          f'rule_order: {len(epo_ro)}, random: {len(epo_rand)}')
    return {'match': epo_match, 'rule_order': epo_ro, 'random': epo_rand}


def select_channels(epochs, ch_list):
    """Restrict epochs to magnetometers in ch_list."""
    epo       = epochs.copy().pick('mag', exclude='bads')
    available = [ch for ch in ch_list if ch in epo.ch_names]
    if not available:
        raise ValueError('None of the requested channels found in epochs.')
    return epo.pick_channels(available)


# ── COHERENCE COMPUTATION ──────────────────────────────────────────────────────

from mne.time_frequency import tfr_array_multitaper

def compute_coherence_pair(epochs, ch_a_list, ch_b_list,
                           freqs, n_cycles, time_bandwidth, decim, label=''):
    print(f'  Coherence [{label}]  n={len(epochs)} trials...')

    all_needed = list(dict.fromkeys(ch_a_list + ch_b_list))
    epo      = epochs.copy().pick_channels(all_needed)
    ch_names = epo.ch_names

    idx_a = [ch_names.index(ch) for ch in ch_a_list if ch in ch_names]
    idx_b = [ch_names.index(ch) for ch in ch_b_list if ch in ch_names]

    data  = epo.get_data()       # (n_epochs, n_ch, n_times)
    sfreq = epo.info['sfreq']

    complex_tfr = tfr_array_multitaper(
        data,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        output='complex',
        decim=decim,
        n_jobs=1,
        verbose=False,
    )
    # complex_tfr shape: (n_epochs, n_ch, n_freqs, n_times_out)

    # Derive times from actual output size — don't rely on epo.times[::decim]
    # which can be off by 1 due to rounding
    n_times_out = complex_tfr.shape[-1]
    times       = np.linspace(epo.tmin, epo.tmax, n_times_out)

    coh_pairs = []
    for ia in idx_a:
        for ib in idx_b:
            x = complex_tfr[:, ia, :, :]   # (n_epochs, n_freqs, n_times_out)
            y = complex_tfr[:, ib, :, :]
            cross   = np.mean(np.conj(x) * y, axis=0)
            power_x = np.mean(np.abs(x) ** 2, axis=0)
            power_y = np.mean(np.abs(y) ** 2, axis=0)
            coh     = np.abs(cross) / np.sqrt(power_x * power_y + 1e-12)
            coh_pairs.append(coh)

    mean_coh = np.mean(coh_pairs, axis=0)   # (n_freqs, n_times_out)
    mean_coh = np.squeeze(mean_coh)
    print(f'    Averaged over {len(coh_pairs)} pairs ({len(idx_a)} x {len(idx_b)})')
    print(f'    Output shape: {mean_coh.shape}, times: {n_times_out} points '
          f'[{times[0]:.3f}, {times[-1]:.3f}] s')
    return mean_coh, times, freqs

# def coherence_difference(coh_a, coh_b, label='A - B'):
#     """Subtract coh_b from coh_a. Both are (n_freqs, n_times) arrays."""
#     return coh_a - coh_b

def coherence_difference(coh_a, coh_b, label='A - B'):
    """Subtract coh_b from coh_a with shape check."""
    if coh_a.shape != coh_b.shape:
        raise ValueError(
            f'Shape mismatch in coherence_difference [{label}]: '
            f'{coh_a.shape} vs {coh_b.shape}'
        )
    return coh_a - coh_b

# ── PERMUTATION TEST ───────────────────────────────────────────────────────────

def permutation_test_coherence(epochs_a, epochs_b,
                                ch_a_list, ch_b_list,
                                freqs, n_cycles, time_bandwidth,
                                decim, n_permutations=100):
    """
    Permutation test for coherence difference between two conditions.
    Identical statistical logic to the ITC permutation test:
      - Pool trials, shuffle labels, recompute coherence difference
      - Z-score = (observed - null_mean) / null_std
      - sig_mask: |Z| > 1.96

    Returns (z_scores_2d, sig_mask) both shape (n_freqs, n_times).
    """
    n_a = len(epochs_a)
    n_b = len(epochs_b)

    # Observed coherence difference
    coh_a, times, _ = compute_coherence_pair(
        epochs_a, ch_a_list, ch_b_list,
        freqs, n_cycles, time_bandwidth, decim, label='perm_obs_a'
    )
    coh_b, _, _ = compute_coherence_pair(
        epochs_b, ch_a_list, ch_b_list,
        freqs, n_cycles, time_bandwidth, decim, label='perm_obs_b'
    )
    observed = coh_a - coh_b   # (n_freqs, n_times)

    # Pool and shuffle
    epochs_pooled = mne.concatenate_epochs([epochs_a, epochs_b])
    null_diffs    = []
    rng           = np.random.default_rng(seed=42)

    for i in range(n_permutations):
        perm         = rng.permutation(n_a + n_b)
        idx_a, idx_b = list(perm[:n_a]), list(perm[n_a:])

        coh_pa, _, _ = compute_coherence_pair(
            epochs_pooled[idx_a], ch_a_list, ch_b_list,
            freqs, n_cycles, time_bandwidth, decim, label=''
        )
        coh_pb, _, _ = compute_coherence_pair(
            epochs_pooled[idx_b], ch_a_list, ch_b_list,
            freqs, n_cycles, time_bandwidth, decim, label=''
        )
        null_diffs.append(coh_pa - coh_pb)

        if (i + 1) % 10 == 0:
            print(f'    Permutation {i+1}/{n_permutations}')

    null_diffs = np.array(null_diffs)   # (n_perm, n_freqs, n_times)
    null_mean  = null_diffs.mean(axis=0)
    null_std   = null_diffs.std(axis=0)
    null_std[null_std == 0] = np.nan

    z_scores = (observed - null_mean) / null_std   # (n_freqs, n_times)
    sig_mask = np.abs(z_scores) > 1.96

    return z_scores, sig_mask, times


# ── PLOTTING ───────────────────────────────────────────────────────────────────

def _crop_2d(data_2d, tmin, tmax, times):
    """Crop a (n_freqs, n_times) array to the plot time window."""
    mask = (times >= tmin) & (times <= tmax)
    return data_2d[:, mask], times[mask]


def draw_box(ax, t0, t1, f0, f1, label, color='black', lw=1.5):
    rect = plt.Rectangle(
        (t0, f0), t1 - t0, f1 - f0,
        linewidth=lw, edgecolor=color, facecolor='none',
        linestyle='--', zorder=5,
    )
    ax.add_patch(rect)
    if label:
        ax.text(t0 + 2, f0 + (f1 - f0) * 0.1, label,
                color='black', fontsize=9, fontweight='bold', zorder=6)


def plot_coherence_differences(diff_ro, diff_rand, times,
                                z_ro=None, z_rand=None,
                                sig_mask_ro=None, sig_mask_rand=None,
                                pair_name='',
                                tmin=TMIN_PLOT, tmax=TMAX_PLOT,
                                alpha_box=ALPHA_BOX, beta_box=BETA_BOX,
                                vlines=VLINES, output_file=None):
    """
    Two-panel figure for one region pair:
      Left  : Coherence(Match) - Coherence(Rule-Order)
      Right : Coherence(Match) - Coherence(Random)

    Plots Z-scores if z_ro/z_rand are provided, otherwise raw difference.
    Black contours at |Z| = 1.96 where sig masks are provided.
    """
    use_zscore = (z_ro is not None) and (z_rand is not None)

    if use_zscore:
        data_ro,   times_plot = _crop_2d(z_ro,   tmin, tmax, times)
        data_rand, _          = _crop_2d(z_rand, tmin, tmax, times)
        cb_label = 'Z-score of Coherence Difference'
    else:
        data_ro,   times_plot = _crop_2d(diff_ro,   tmin, tmax, times)
        data_rand, _          = _crop_2d(diff_rand, tmin, tmax, times)
        cb_label = 'Coherence Difference (Match - Other)'

    if sig_mask_ro is not None:
        sig_ro_plot,   _ = _crop_2d(sig_mask_ro.astype(float),   tmin, tmax, times)
        sig_rand_plot, _ = _crop_2d(sig_mask_rand.astype(float), tmin, tmax, times)
    else:
        sig_ro_plot = sig_rand_plot = None

    vmax = max(np.abs(data_ro).max(), np.abs(data_rand).max())
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = 'RdBu_r'
    t_ms = 1000

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(
        1, 3, width_ratios=[1, 1, 0.05],
        left=0.07, right=0.93, bottom=0.12, top=0.88, wspace=0.28,
    )
    ax_ro   = fig.add_subplot(gs[0, 0])
    ax_rand = fig.add_subplot(gs[0, 1])
    ax_cb   = fig.add_subplot(gs[0, 2])

    panels = [
        (ax_ro,   data_ro,   sig_ro_plot,   'Coherence  Match - Rule-Order'),
        (ax_rand, data_rand, sig_rand_plot, 'Coherence  Match - Random'),
    ]

    for ax, data, sig_plot, title in panels:
        ax.pcolormesh(times_plot * t_ms, FREQS, data,
                      norm=norm, cmap=cmap, shading='auto')
        if sig_plot is not None:
            ax.contour(times_plot * t_ms, FREQS, sig_plot,
                       levels=[0.5], colors='black', linewidths=1.0)
        # draw_box(ax, alpha_box[0] * t_ms, alpha_box[1] * t_ms,
        #          alpha_box[2], alpha_box[3], label='Alpha')
        # draw_box(ax, beta_box[0]  * t_ms, beta_box[1]  * t_ms,
        #          beta_box[2],  beta_box[3],  label='Beta')
        for i, vt in enumerate(vlines):
            kw = dict(color='red', lw=1.8, ls='-') if i == 0 \
                 else dict(color='limegreen', lw=1.4, ls='--')
            ax.axvline(vt * t_ms, **kw)
        ax.set_xlim(tmin * t_ms, tmax * t_ms)
        ax.set_ylim(FREQS[0], FREQS[-1])
        ax.set_xlabel('Time since Onset of Stimulus (ms)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb
    ).set_label(cb_label, fontsize=9)

    title_str = 'Inter-Region Spectral Coherence Difference'
    if pair_name:
        title_str += f'  [{pair_name}]'
    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=0.96)

    if output_file is not None:
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f'  Figure saved -> {output_file}')
    else:
        plt.show()
    return fig


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def run_pipeline():
    # 1. Load and label conditions (identical to ITC pipeline)
    print('Loading epochs and behavioural data...')
    cond_epochs = load_and_label(EPOCHS_FILE, BEH_FILE)

    # 2. Get full magnetometer channel list
    all_ch = cond_epochs['match'].copy().pick('mag', exclude='bads').ch_names

    # 3. Build region -> channel list mapping
    region_channels = {}
    print('\nChannel counts per region:')
    for region, patterns in REGIONS.items():
        ch_list = channels_for_region(all_ch, patterns, ch_suffix=CH_SUFFIX)
        region_channels[region] = ch_list
        print(f'  {region:12s}: {len(ch_list)} channels  (e.g. {ch_list[:3]})')

    all_results = {}

    # 4. Run coherence pipeline for each region pair
    for region_a, region_b in REGION_PAIRS:
        pair_name = f'{region_a}-{region_b}'
        ch_a = region_channels[region_a]
        ch_b = region_channels[region_b]

        sep = '=' * 50
        print(f'\n{sep}')
        print(f'  PAIR: {pair_name.upper()}')
        print(f'  {region_a}: {len(ch_a)} ch  x  {region_b}: {len(ch_b)} ch'
              f'  =  {len(ch_a) * len(ch_b)} pairs')
        print(sep)

        if not ch_a or not ch_b:
            print(f'  Skipping {pair_name} — one region has no channels.')
            continue

        # Select epochs to only the channels needed for this pair
        needed = list(dict.fromkeys(ch_a + ch_b))
        epochs_match = select_channels(cond_epochs['match'],      needed)
        epochs_ro    = select_channels(cond_epochs['rule_order'], needed)
        epochs_rand  = select_channels(cond_epochs['random'],     needed)

        # Coherence per condition
        print('  Computing observed coherence...')
        coh_match, times, _ = compute_coherence_pair(
            epochs_match, ch_a, ch_b,
            FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM, f'match/{pair_name}'
        )
        coh_ro, _, _ = compute_coherence_pair(
            epochs_ro, ch_a, ch_b,
            FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM, f'rule_order/{pair_name}'
        )
        coh_rand, _, _ = compute_coherence_pair(
            epochs_rand, ch_a, ch_b,
            FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM, f'random/{pair_name}'
        )

        # Difference maps
        print(f'DEBUG coh_match: {coh_match.shape}, times: {times.shape}')
        print(f'DEBUG coh_ro:    {coh_ro.shape}')
        print(f'DEBUG coh_rand:  {coh_rand.shape}')

        diff_ro   = coherence_difference(coh_match, coh_ro,   f'Match-RuleOrder/{pair_name}')
        diff_rand = coherence_difference(coh_match, coh_rand, f'Match-Random/{pair_name}')

        print(f'DEBUG diff_ro:   {diff_ro.shape}')
        print(f'DEBUG times:     {times.shape}')

        # Permutation tests
        z_ro = z_rand = sig_mask_ro = sig_mask_rand = None
        if N_PERMUTATIONS > 0:
            print('  Permutation test: match vs rule_order...')
            z_ro, sig_mask_ro, _ = permutation_test_coherence(
                epochs_match, epochs_ro,
                ch_a, ch_b,
                FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM, N_PERMUTATIONS,
            )
            print('  Permutation test: match vs random...')
            z_rand, sig_mask_rand, _ = permutation_test_coherence(
                epochs_match, epochs_rand,
                ch_a, ch_b,
                FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM, N_PERMUTATIONS,
            )

        # Plot and save
        out_file = OUTPUT_DIR / f'coherence_{pair_name}_perm.png'
        fig = plot_coherence_differences(
            diff_ro, diff_rand, times,
            z_ro=z_ro if N_PERMUTATIONS > 0 else None,
            z_rand=z_rand if N_PERMUTATIONS > 0 else None,
            sig_mask_ro=sig_mask_ro,
            sig_mask_rand=sig_mask_rand,
            pair_name=pair_name,
            output_file=out_file,
        )

        all_results[pair_name] = dict(
            coh_match=coh_match, coh_ro=coh_ro, coh_rand=coh_rand,
            diff_ro=diff_ro, diff_rand=diff_rand,
            z_ro=z_ro, z_rand=z_rand,
            sig_mask_ro=sig_mask_ro, sig_mask_rand=sig_mask_rand,
            times=times, fig=fig,
        )

    print('\nDone. Output files:')
    for pair_name in all_results:
        print(f'  {OUTPUT_DIR}/coherence_{pair_name}.png')

    return all_results


if __name__ == '__main__':
    results = run_pipeline()