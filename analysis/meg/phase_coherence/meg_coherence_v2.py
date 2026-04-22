"""
Phase-Locked Coherence Difference Pipeline
===========================================

Loads epochs and a behavioural CSV, splits test_non_match trials into
rule_order vs random subtypes, then computes ITC and plots:
  • ITC(Match) − ITC(Rule-Order Non-Match)
  • ITC(Match) − ITC(Random Non-Match)

as time-frequency maps using multitaper wavelets (better suited to short
epochs than Morlet).

Usage
-----
1. Set EPOCHS_FILE and BEH_FILE to your *_epo.fif and *_events.csv paths.
2. Run:  python phase_locked_coherence_pipeline.py
         or execute cell-by-cell in Jupyter.

Dependencies
------------
  mne >= 1.6, numpy, pandas, matplotlib
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import mne

mne.set_log_level('WARNING')

# ── USER SETTINGS ─────────────────────────────────────────────────────────────

EPOCHS_FILE = pathlib.Path('/jukebox/PNI-classes/students/NEU502/2026-NEU502B/binding/data/binding/bids/derivatives/analysis1__1/sub-001/ses-01/meg/sub-001_ses-01_task-binding_proc-clean_epo.fif')
BEH_FILE    = pathlib.Path('/usr/people/ao8210/neu502b/brain-binding/data/2026-04-10/sub-001_events.csv')

# Channels: set to None to use all mag channels, or give a list of names
CHANNEL_PICKS = None    # e.g. ['P5 36 Z', 'P6 10 Z']
CH_SUFFIX     = None   # radial Z channels only; set to None to use all mag

# TFR parameters (multitaper — better for short 500 ms epochs)
FREQS         = np.arange(8, 41, 1)   # 8–40 Hz
N_CYCLES      = FREQS / 4.0           # short wavelets safe for 201 samples
TIME_BANDWIDTH = 2.0                  # spectral smoothing (increase → smoother)
DECIM         = 1                     # no decimation on short epochs

# Plot limits (seconds, converted to ms on axis)
TMIN_PLOT = -0.100
TMAX_PLOT =  0.400

# Highlight boxes: (t_start_s, t_end_s, f_low_hz, f_high_hz)
ALPHA_BOX = (-0.05, 0.150,  8, 12)
BETA_BOX  = (0.0, 0.250, 13, 30)

# Vertical reference lines (seconds)
VLINES = [0.0]   # stimulus onset; add more if needed e.g. [0.0, 0.175, 0.225]

# Output file — set to None to display interactively
OUTPUT_FILE = pathlib.Path('./itc_difference_plot_perm.png')
DPI = 150

# ── END USER SETTINGS ─────────────────────────────────────────────────────────


def load_and_label(epochs_path: str, beh_path: str) -> dict:
    """
    Load epochs and split test_non_match into rule_order / random subtypes
    using the behavioural CSV.

    Returns a dict mapping condition name → Epochs object:
      'match'      — test_match trials
      'rule_order' — test_non_match where mismatch_type == 'rule_order'
      'random'     — test_non_match where mismatch_type == 'random'
    """
    epo = mne.read_epochs(epochs_path, preload=True, verbose=False)
    df  = pd.read_csv(beh_path).sort_values('trial').reset_index(drop=True)

    # ── 1. Isolate test rows from metadata ────────────────────────────────────
    meta      = epo.metadata.copy()
    test_mask = meta['event_name'].isin(['test_match', 'test_non_match'])
    meta_test = meta[test_mask]   # 120 rows, in epoch order

    # ── 2. Verify match/non-match sequence agrees with CSV ────────────────────
    epo_seq = (meta_test['event_name'] == 'test_match').astype(int).values
    beh_seq = df['match'].values
    if not np.array_equal(epo_seq, beh_seq):
        raise ValueError(
            'Match/non-match sequence in epochs does not match behavioural CSV. '
            'Cannot safely cross-reference by trial order.'
        )

    # ── 3. Propagate mismatch_type into metadata ──────────────────────────────
    meta['mismatch_type'] = ''
    meta.loc[meta_test.index, 'mismatch_type'] = (
        df['mismatch_type'].fillna('').values
    )
    epo.metadata = meta

    # ── 4. Build per-condition epochs ─────────────────────────────────────────
    epo_match = epo['test_match']
    epo_ro    = epo[epo.metadata['mismatch_type'] == 'rule_order']
    epo_rand  = epo[epo.metadata['mismatch_type'] == 'random']

    print(
        f'Epoch counts — match: {len(epo_match)}, '
        f'rule_order: {len(epo_ro)}, random: {len(epo_rand)}'
    )
    return {'match': epo_match, 'rule_order': epo_ro, 'random': epo_rand}


def select_channels(epochs, ch_suffix=CH_SUFFIX, picks=CHANNEL_PICKS):
    """Pick magnetometer Z channels (radial component)."""
    epo = epochs.copy().pick('mag', exclude='bads')
    if ch_suffix is not None:
        z_names = [ch for ch in epo.ch_names if ch.endswith(ch_suffix)]
        epo = epo.pick_channels(z_names)
        print(f'Z channels selected : {len(z_names)}')
    if picks is not None:
        epo = epo.pick_channels(picks)
        print(f'Restricted to {len(picks)} user-specified channels')
    return epo


def compute_itc(epochs_cond, freqs, n_cycles, time_bandwidth, decim, label=''):
    """Compute ITC via multitaper (discards power). Returns AverageTFR."""
    print(f'  Computing ITC [{label}]  n={len(epochs_cond)} trials…')
    _, itc = mne.time_frequency.tfr_multitaper(
        epochs_cond,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        use_fft=True,
        return_itc=True,
        decim=decim,
        n_jobs=1,
        verbose=False,
    )
    return itc


def itc_difference(itc_a, itc_b, label='A − B'):
    """Subtract itc_b from itc_a channel-wise; return a new AverageTFR."""
    diff = itc_a.copy()
    diff.data    = itc_a.data - itc_b.data
    diff.comment = f'ITC: {label}'
    return diff


def average_over_channels(tfr_obj, tmin=None, tmax=None):
    """Average across channels, optionally crop time. Returns (times, freqs, data)."""
    data  = tfr_obj.data.mean(axis=0)   # (n_freqs, n_times)
    times = tfr_obj.times.copy()
    if tmin is not None:
        mask  = (times >= tmin) & (times <= tmax)
        times = times[mask]
        data  = data[:, mask]
    return times, tfr_obj.freqs, data


def draw_box(ax, t0, t1, f0, f1, label, color='black', lw=1.5):
    """Overlay a dashed rectangle on a TFR axis."""
    rect = plt.Rectangle(
        (t0, f0), t1 - t0, f1 - f0,
        linewidth=lw, edgecolor=color, facecolor='none',
        linestyle='--', zorder=5,
    )
    ax.add_patch(rect)
    if label:
        ax.text(t0 + 2, f0 + (f1 - f0) * 0.1, label,
                color='black', fontsize=9, fontweight='bold', zorder=6)
        

def permutation_test_itc(epochs_a, epochs_b, freqs, n_cycles, time_bandwidth,
                          decim, n_permutations=1000, n_jobs=-1):
    """
    Permutation test for ITC difference between two conditions.
    Returns observed difference, p-values, and significance mask.
    """
    n_a = len(epochs_a)
    n_b = len(epochs_b)

    # 1. Observed difference
    _, itc_a = mne.time_frequency.tfr_multitaper(
        epochs_a, freqs=freqs, n_cycles=n_cycles,
        time_bandwidth=time_bandwidth, return_itc=True,
        decim=decim, n_jobs=1, verbose=False,
    )
    _, itc_b = mne.time_frequency.tfr_multitaper(
        epochs_b, freqs=freqs, n_cycles=n_cycles,
        time_bandwidth=time_bandwidth, return_itc=True,
        decim=decim, n_jobs=1, verbose=False,
    )
    observed = itc_a.data - itc_b.data   # (n_channels, n_freqs, n_times)

    # 2. Pool all epochs together
    epochs_pooled = mne.concatenate_epochs([epochs_a, epochs_b])

    # 3. Build null distribution by shuffling labels
    null_diffs = []
    rng = np.random.default_rng(seed=42)

    for i in range(n_permutations):
        perm = rng.permutation(n_a + n_b)
        idx_a, idx_b = perm[:n_a], perm[n_a:]

        _, itc_perm_a = mne.time_frequency.tfr_multitaper(
            epochs_pooled[idx_a], freqs=freqs, n_cycles=n_cycles,
            time_bandwidth=time_bandwidth, return_itc=True,
            decim=decim, n_jobs=1, verbose=False,
        )
        _, itc_perm_b = mne.time_frequency.tfr_multitaper(
            epochs_pooled[idx_b], freqs=freqs, n_cycles=n_cycles,
            time_bandwidth=time_bandwidth, return_itc=True,
            decim=decim, n_jobs=1, verbose=False,
        )
        null_diffs.append(itc_perm_a.data - itc_perm_b.data)
        
        if (i + 1) % 100 == 0:
            print(f'  Permutation {i+1}/{n_permutations}')

    null_diffs = np.array(null_diffs)  # (n_permutations, n_channels, n_freqs, n_times)

    # 4. p-value: proportion of null differences >= observed (one-tailed)
    #    Use two-tailed: proportion of |null| >= |observed|
    p_values = (np.abs(null_diffs) >= np.abs(observed)).mean(axis=0)

    # 5. Significance mask (after averaging across channels for plotting)
    p_avg    = p_values.mean(axis=0)        # (n_freqs, n_times)
    sig_mask = p_avg < 0.05                 # uncorrected

    return observed, p_values, sig_mask

def plot_itc_differences(diff_ro, diff_rand,
                         tmin=TMIN_PLOT, tmax=TMAX_PLOT,
                         alpha_box=ALPHA_BOX, beta_box=BETA_BOX,
                         vlines=VLINES, output_file=OUTPUT_FILE):
    """
    Two-panel figure:
      Left  : ITC(Match) − ITC(Rule-Order)
      Right : ITC(Match) − ITC(Random)
    """
    times_ro,   freqs_ro,   data_ro   = average_over_channels(diff_ro,   tmin, tmax)
    times_rand, freqs_rand, data_rand = average_over_channels(diff_rand, tmin, tmax)

    # Symmetric colour scale centred on zero
    vmax = max(np.abs(data_ro).max(), np.abs(data_rand).max())
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = 'RdBu_r'
    t_ms = 1000   # seconds → ms

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(
        1, 3, width_ratios=[1, 1, 0.05],
        left=0.07, right=0.93, bottom=0.12, top=0.88, wspace=0.28,
    )
    ax_ro   = fig.add_subplot(gs[0, 0])
    ax_rand = fig.add_subplot(gs[0, 1])
    ax_cb   = fig.add_subplot(gs[0, 2])

    panels = [
        (ax_ro,   times_ro,   freqs_ro,   data_ro,   'ITC  Match − NonMatch'),
        (ax_rand, times_rand, freqs_rand, data_rand, 'ITC  Match − Random'),
    ]

    for ax, times, freqs, data, title in panels:
        ax.pcolormesh(times * t_ms, freqs, data,
                      norm=norm, cmap=cmap, shading='auto')
        
        ax.contour(
            times * t_ms, freqs, sig_mask.astype(float),
            levels=[0.5], colors='black', linewidths=1.0,
        )

        draw_box(ax,
                 alpha_box[0] * t_ms, alpha_box[1] * t_ms,
                 alpha_box[2], alpha_box[3], label="'Alpha'")
        draw_box(ax,
                 beta_box[0]  * t_ms, beta_box[1]  * t_ms,
                 beta_box[2],  beta_box[3],  label="'Beta'")

        for i, vt in enumerate(vlines):
            kw = dict(color='red', lw=1.8, ls='-') if i == 0 \
                 else dict(color='limegreen', lw=1.4, ls='--')
            ax.axvline(vt * t_ms, **kw)

        ax.set_xlim(tmin * t_ms, tmax * t_ms)
        ax.set_ylim(freqs[0], freqs[-1])
        ax.set_xlabel('Time since Onset of Stimulus (ms)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb
    ).set_label('ITC Difference\n(Match − Other)', fontsize=9)

    fig.suptitle('Phase-Locked Coherence Difference', fontsize=13,
                 fontweight='bold', y=0.96)

    if output_file is not None:
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f'\nFigure saved → {output_file}')
    else:
        plt.show()

    return fig


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline():
    # 1. Load and label conditions from epochs + behavioural CSV
    print('Loading epochs and behavioural data…')
    cond_epochs = load_and_label(EPOCHS_FILE, BEH_FILE)

    # 2. Select Z channels for each condition
    print('\nSelecting Z channels…')
    epochs_match = select_channels(cond_epochs['match'])
    epochs_ro    = select_channels(cond_epochs['rule_order'])
    epochs_rand  = select_channels(cond_epochs['random'])

    # 3. Compute ITC per condition
    print('\nComputing ITC…')
    itc_match = compute_itc(epochs_match, FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM, 'match')
    itc_ro    = compute_itc(epochs_ro,    FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM, 'rule_order')
    itc_rand  = compute_itc(epochs_rand,  FREQS, N_CYCLES, TIME_BANDWIDTH, DECIM, 'random')

    # 4. Difference maps
    print('\nComputing differences…')
    diff_ro   = itc_difference(itc_match, itc_ro,   'Match − Rule-Order')
    diff_rand = itc_difference(itc_match, itc_rand, 'Match − Random')

    print(f'  Match−RuleOrder range : [{diff_ro.data.min():.4f}, {diff_ro.data.max():.4f}]')
    print(f'  Match−Random    range : [{diff_rand.data.min():.4f}, {diff_rand.data.max():.4f}]')

    # 5. Plot
    print('\nPlotting…')
    fig = plot_itc_differences(diff_ro, diff_rand)

    return dict(
        epochs_match=epochs_match,
        epochs_ro=epochs_ro,
        epochs_rand=epochs_rand,
        itc_match=itc_match,
        itc_ro=itc_ro,
        itc_rand=itc_rand,
        diff_ro=diff_ro,
        diff_rand=diff_rand,
        fig=fig,
    )


if __name__ == '__main__':
    results = run_pipeline()