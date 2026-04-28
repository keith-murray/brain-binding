"""
Coherence by Condition Pipeline
=================================
Plots raw coherence / ITC for each of the three trial types:
  match, rule_order, random

Two analyses, one script:
  1. Within-region ITC  — one figure per region (occipital, parietal, frontal)
     Each figure has 3 panels: match | rule_order | random
  2. Inter-region coherence — one figure per region pair
     Each figure has 3 panels: match | rule_order | random

No permutation testing (N_PERMUTATIONS = 0).

Outputs
-------
  itc_bycond_occipital.png
  itc_bycond_parietal.png
  itc_bycond_frontal.png
  coherence_bycond_occipital-parietal.png
  coherence_bycond_occipital-frontal.png
  coherence_bycond_parietal-frontal.png
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
from mne.time_frequency import tfr_array_multitaper

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

REGIONS = {
    'occipital' : [r'^O', r'^Iz'],
    'parietal'  : [r'^P'],
    'frontal'   : [r'^F[^p]', r'^AF'],
}

REGION_PAIRS = list(itertools.combinations(REGIONS.keys(), 2))

CH_SUFFIX = None

FREQS          = np.arange(8, 41, 1)
N_CYCLES       = FREQS / 4.0
TIME_BANDWIDTH = 2.0
DECIM          = 1

TMIN_PLOT = -0.100
TMAX_PLOT =  0.400

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

_args       = _parse_args()
EPOCHS_FILE = _args.epochs    or EPOCHS_FILE
BEH_FILE    = _args.beh       or BEH_FILE
OUTPUT_DIR  = _args.outputdir or OUTPUT_DIR

# Condition display labels
COND_LABELS = {
    'match'      : 'Match',
    'rule_order' : 'Rule-Order',
    'random'     : 'Random',
}


# ── CHANNEL SELECTION ──────────────────────────────────────────────────────────

def channels_for_region(all_ch_names, patterns, ch_suffix=None):
    matched = [ch for ch in all_ch_names
               if any(re.match(pat, ch) for pat in patterns)]
    if ch_suffix is not None:
        matched = [ch for ch in matched if ch.endswith(ch_suffix)]
    return matched


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_and_label(epochs_path, beh_path):
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
    epo       = epochs.copy().pick('mag', exclude='bads')
    available = [ch for ch in ch_list if ch in epo.ch_names]
    if not available:
        raise ValueError('None of the requested channels found in epochs.')
    return epo.pick_channels(available)


# ── WITHIN-REGION ITC ─────────────────────────────────────────────────────────

def compute_itc(epochs_cond, freqs, n_cycles, time_bandwidth, decim, label=''):
    """Compute ITC via multitaper. Returns (itc_2d, times) averaged over channels."""
    print(f'  ITC [{label}]  n={len(epochs_cond)} trials...')
    _, itc_tfr = mne.time_frequency.tfr_multitaper(
        epochs_cond, freqs=freqs, n_cycles=n_cycles,
        time_bandwidth=time_bandwidth, use_fft=True,
        return_itc=True, decim=decim, n_jobs=1, verbose=False,
    )
    # itc_tfr.data shape: (n_ch, n_freqs, n_times)
    itc_2d = itc_tfr.data.mean(axis=0)   # (n_freqs, n_times)
    times  = itc_tfr.times
    return itc_2d, times


# ── INTER-REGION COHERENCE ────────────────────────────────────────────────────

def compute_coherence_pair(epochs, ch_a_list, ch_b_list,
                           freqs, n_cycles, time_bandwidth, decim, label=''):
    """Compute mean pairwise coherence between two channel sets.
    Returns (coh_2d, times) shape (n_freqs, n_times)."""
    print(f'  Coherence [{label}]  n={len(epochs)} trials...')

    all_needed = list(dict.fromkeys(ch_a_list + ch_b_list))
    epo        = epochs.copy().pick_channels(all_needed)
    ch_names   = epo.ch_names
    idx_a      = [ch_names.index(ch) for ch in ch_a_list if ch in ch_names]
    idx_b      = [ch_names.index(ch) for ch in ch_b_list if ch in ch_names]

    data  = epo.get_data()
    sfreq = epo.info['sfreq']

    complex_tfr = tfr_array_multitaper(
        data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
        time_bandwidth=time_bandwidth, output='complex',
        decim=decim, n_jobs=1, verbose=False,
    )
    # complex_tfr: (n_epochs, n_ch, n_freqs, n_times_out)

    n_times_out = complex_tfr.shape[-1]
    times       = np.linspace(epo.tmin, epo.tmax, n_times_out)

    coh_pairs = []
    for ia in idx_a:
        for ib in idx_b:
            x = complex_tfr[:, ia, :, :]
            y = complex_tfr[:, ib, :, :]
            cross   = np.mean(np.conj(x) * y, axis=0).squeeze()
            power_x = np.mean(np.abs(x) ** 2, axis=0).squeeze()
            power_y = np.mean(np.abs(y) ** 2, axis=0).squeeze()
            coh     = np.abs(cross) / np.sqrt(power_x * power_y + 1e-12)
            coh_pairs.append(coh)

    mean_coh = np.mean(coh_pairs, axis=0).squeeze()   # (n_freqs, n_times)
    print(f'    {len(coh_pairs)} pairs ({len(idx_a)} x {len(idx_b)}), '
          f'shape: {mean_coh.shape}')
    return mean_coh, times


# ── SHARED PLOTTING ───────────────────────────────────────────────────────────

def _crop_2d(data_2d, tmin, tmax, times):
    """Crop (n_freqs, n_times) to plot window."""
    mask = (times >= tmin) & (times <= tmax)
    return data_2d[:, mask], times[mask]


def plot_three_conditions(data_dict, times_dict, freqs,
                          title, cb_label,
                          tmin=TMIN_PLOT, tmax=TMAX_PLOT,
                          vlines=VLINES, output_file=None):
    """
    Three-panel figure: match | rule_order | random.

    data_dict  : {'match': 2d_array, 'rule_order': 2d_array, 'random': 2d_array}
    times_dict : same keys, each a 1d times array
    freqs      : 1d frequency array
    """
    conds = ['match', 'rule_order', 'random']

    # Crop all to plot window
    cropped = {}
    for cond in conds:
        d, t = _crop_2d(data_dict[cond], tmin, tmax, times_dict[cond])
        cropped[cond] = (d, t)

    # Shared colour scale across all three panels
    vmax = max(np.abs(cropped[c][0]).max() for c in conds)
    vmin = 0.0   # ITC and coherence are non-negative
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = 'hot_r'
    t_ms = 1000

    fig = plt.figure(figsize=(17, 5))
    gs  = gridspec.GridSpec(
        1, 4, width_ratios=[1, 1, 1, 0.05],
        left=0.06, right=0.93, bottom=0.12, top=0.85, wspace=0.25,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cb = fig.add_subplot(gs[0, 3])

    for ax, cond in zip(axes, conds):
        data, times_plot = cropped[cond]
        ax.pcolormesh(times_plot * t_ms, freqs, data,
                      norm=norm, cmap=cmap, shading='auto')
        for i, vt in enumerate(vlines):
            kw = dict(color='red', lw=1.8, ls='-') if i == 0 \
                 else dict(color='limegreen', lw=1.4, ls='--')
            ax.axvline(vt * t_ms, **kw)
        ax.set_xlim(tmin * t_ms, tmax * t_ms)
        ax.set_ylim(freqs[0], freqs[-1])
        ax.set_xlabel('Time since Onset of Stimulus (ms)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        ax.set_title(COND_LABELS[cond], fontsize=11, fontweight='bold', pad=8)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb
    ).set_label(cb_label, fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.97)

    if output_file is not None:
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f'  Saved -> {output_file}')
    else:
        plt.show()

    plt.close(fig)
    return fig


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline():
    print('Loading epochs and behavioural data...')
    cond_epochs = load_and_label(EPOCHS_FILE, BEH_FILE)

    all_ch = cond_epochs['match'].copy().pick('mag', exclude='bads').ch_names

    region_channels = {}
    print('\nChannel counts per region:')
    for region, patterns in REGIONS.items():
        ch_list = channels_for_region(all_ch, patterns, ch_suffix=CH_SUFFIX)
        region_channels[region] = ch_list
        print(f'  {region:12s}: {len(ch_list)} channels')

    all_results = {}

    # ── 1. WITHIN-REGION ITC ─────────────────────────────────────────────────

    print('\n' + '=' * 55)
    print('  WITHIN-REGION ITC')
    print('=' * 55)

    for region, ch_list in region_channels.items():
        print(f'\n  Region: {region.upper()}  ({len(ch_list)} channels)')

        if not ch_list:
            continue

        itc_data  = {}
        itc_times = {}

        for cond in ['match', 'rule_order', 'random']:
            epo_sel = select_channels(cond_epochs[cond], ch_list)
            itc_2d, times = compute_itc(
                epo_sel, FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM,
                label=f'{cond}/{region}'
            )
            itc_data[cond]  = itc_2d
            itc_times[cond] = times

        out_file = OUTPUT_DIR / f'itc_bycond_{region}.png'
        fig = plot_three_conditions(
            itc_data, itc_times, FREQS,
            title=f'Within-Region ITC  [{region.capitalize()}]',
            cb_label='Inter-Trial Coherence',
            output_file=out_file,
        )

        all_results[f'itc_{region}'] = dict(
            itc_data=itc_data, itc_times=itc_times, fig=fig
        )

    # ── 2. INTER-REGION COHERENCE ─────────────────────────────────────────────

    print('\n' + '=' * 55)
    print('  INTER-REGION COHERENCE')
    print('=' * 55)

    for region_a, region_b in REGION_PAIRS:
        pair_name = f'{region_a}-{region_b}'
        ch_a = region_channels[region_a]
        ch_b = region_channels[region_b]

        print(f'\n  Pair: {pair_name.upper()}  '
              f'({len(ch_a)} x {len(ch_b)} = {len(ch_a)*len(ch_b)} pairs)')

        if not ch_a or not ch_b:
            print(f'  Skipping — one region has no channels.')
            continue

        needed = list(dict.fromkeys(ch_a + ch_b))

        coh_data  = {}
        coh_times = {}

        for cond in ['match', 'rule_order', 'random']:
            epo_sel = select_channels(cond_epochs[cond], needed)
            coh_2d, times = compute_coherence_pair(
                epo_sel, ch_a, ch_b,
                FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM,
                label=f'{cond}/{pair_name}'
            )
            coh_data[cond]  = coh_2d
            coh_times[cond] = times

        out_file = OUTPUT_DIR / f'coherence_bycond_{pair_name}.png'
        fig = plot_three_conditions(
            coh_data, coh_times, FREQS,
            title=f'Inter-Region Coherence  [{pair_name.replace("-", " - ").title()}]',
            cb_label='Spectral Coherence',
            output_file=out_file,
        )

        all_results[f'coherence_{pair_name}'] = dict(
            coh_data=coh_data, coh_times=coh_times, fig=fig
        )

    print('\nDone. Output files:')
    for key in all_results:
        prefix = 'itc_bycond' if key.startswith('itc') else 'coherence_bycond'
        name   = key.replace('itc_', '').replace('coherence_', '')
        print(f'  {OUTPUT_DIR}/{prefix}_{name}.png')

    return all_results


if __name__ == '__main__':
    results = run_pipeline()