"""
Phase-Locked Coherence Difference Pipeline
===========================================
Loads epochs and a behavioural CSV, splits test_non_match trials into
rule_order vs random subtypes, computes ITC per region (occipital, parietal,
frontal) and plots ITC(Match) - ITC(Rule-Order) and ITC(Match) - ITC(Random)
for each region with optional permutation-test significance contours.

Outputs one PNG per region:
  itc_difference_occipital.png
  itc_difference_parietal.png
  itc_difference_frontal.png
"""

import argparse
import re
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

# Region definitions: keys are output labels, values are lists of regex patterns
# matched against channel names (10-20 position prefix).
# Occipital: O*, Iz  |  Parietal: P*  |  Frontal: F* excl Fp, AF*
REGIONS = {
    'occipital' : [r'^O', r'^Iz'],
    'parietal'  : [r'^P'],
    'frontal'   : [r'^F[^p]', r'^AF'],
}

# None = use all axes (X, Y, Z) — recommended for sulcal sources such as
# visual cortex (calcarine sulcus) and IPS.
# Set to ' Z' to restrict to radial channels only.
CH_SUFFIX = None

# TFR parameters (multitaper — better than Morlet for short 500 ms epochs)
FREQS          = np.arange(8, 41, 1)   # 8-40 Hz
N_CYCLES       = FREQS / 4.0           # short wavelets, safe for 201 samples
TIME_BANDWIDTH = 2.0                   # spectral smoothing
DECIM          = 1                     # no decimation on short epochs

# Set to 0 to skip permutation testing and plot without significance contours
N_PERMUTATIONS = 0

# Plot limits (seconds, converted to ms on axis)
TMIN_PLOT = -0.100
TMAX_PLOT =  0.400

# Highlight boxes: (t_start_s, t_end_s, f_low_hz, f_high_hz)
ALPHA_BOX = (0,0,0,0)
BETA_BOX  = (0,0,0,0)

# Vertical reference lines (seconds); first is drawn in red (stimulus onset)
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

_args = _parse_args()
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
    """Load epochs and split test_non_match into rule_order / random subtypes."""
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


# ── TFR COMPUTATION ────────────────────────────────────────────────────────────

def compute_itc(epochs_cond, freqs, n_cycles, time_bandwidth, decim, label=''):
    """Compute ITC via multitaper. Returns AverageTFR."""
    print(f'  ITC [{label}]  n={len(epochs_cond)} trials...')
    _, itc = mne.time_frequency.tfr_multitaper(
        epochs_cond, freqs=freqs, n_cycles=n_cycles,
        time_bandwidth=time_bandwidth, use_fft=True,
        return_itc=True, decim=decim, n_jobs=1, verbose=False,
    )
    return itc


def itc_difference(itc_a, itc_b, label='A - B'):
    """Return itc_a - itc_b as a new AverageTFR."""
    diff         = itc_a.copy()
    diff.data    = itc_a.data - itc_b.data
    diff.comment = f'ITC: {label}'
    return diff


# ── PERMUTATION TEST ───────────────────────────────────────────────────────────

# def permutation_test_itc(epochs_a, epochs_b, freqs, n_cycles, time_bandwidth,
#                           decim, n_permutations=1000):
#     """
#     Two-tailed permutation test on ITC difference.
#     Returns (observed, p_values, sig_mask) where sig_mask is (n_freqs, n_times),
#     True where p < 0.05 (uncorrected, averaged across channels).
#     """
#     n_a = len(epochs_a)
#     n_b = len(epochs_b)

#     _, itc_a = mne.time_frequency.tfr_multitaper(
#         epochs_a, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth,
#         return_itc=True, decim=decim, n_jobs=1, verbose=False,
#     )
#     _, itc_b = mne.time_frequency.tfr_multitaper(
#         epochs_b, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth,
#         return_itc=True, decim=decim, n_jobs=1, verbose=False,
#     )
#     observed      = itc_a.data - itc_b.data   # (n_ch, n_freqs, n_times)
#     epochs_pooled = mne.concatenate_epochs([epochs_a, epochs_b])
#     null_diffs    = []
#     rng           = np.random.default_rng(seed=42)

#     for i in range(n_permutations):
#         perm         = rng.permutation(n_a + n_b)
#         idx_a, idx_b = perm[:n_a], perm[n_a:]
#         _, itc_pa = mne.time_frequency.tfr_multitaper(
#             epochs_pooled[idx_a], freqs=freqs, n_cycles=n_cycles,
#             time_bandwidth=time_bandwidth, return_itc=True,
#             decim=decim, n_jobs=1, verbose=False,
#         )
#         _, itc_pb = mne.time_frequency.tfr_multitaper(
#             epochs_pooled[idx_b], freqs=freqs, n_cycles=n_cycles,
#             time_bandwidth=time_bandwidth, return_itc=True,
#             decim=decim, n_jobs=1, verbose=False,
#         )
#         null_diffs.append(itc_pa.data - itc_pb.data)
#         if (i + 1) % 100 == 0:
#             print(f'    Permutation {i+1}/{n_permutations}')

#     null_diffs = np.array(null_diffs)   # (n_perm, n_ch, n_freqs, n_times)
#     p_values   = (np.abs(null_diffs) >= np.abs(observed)).mean(axis=0)
#     sig_mask   = p_values.mean(axis=0) < 0.05   # (n_freqs, n_times)
#     return observed, p_values, sig_mask

def permutation_test_itc(epochs_a, epochs_b, freqs, n_cycles, time_bandwidth,
                          decim, n_permutations=1000):
    """
    Permutation test returning Z-scores instead of p-values.
    Z > 1.96  : condition A significantly greater than B
    Z < -1.96 : condition B significantly greater than A
    """
    n_a = len(epochs_a)
    n_b = len(epochs_b)

    _, itc_a = mne.time_frequency.tfr_multitaper(
        epochs_a, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth,
        return_itc=True, decim=decim, n_jobs=1, verbose=False,
    )
    _, itc_b = mne.time_frequency.tfr_multitaper(
        epochs_b, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth,
        return_itc=True, decim=decim, n_jobs=1, verbose=False,
    )
    observed      = itc_a.data - itc_b.data   # (n_ch, n_freqs, n_times)
    epochs_pooled = mne.concatenate_epochs([epochs_a, epochs_b])
    null_diffs    = []
    rng           = np.random.default_rng(seed=42)

    for i in range(n_permutations):
        perm         = rng.permutation(n_a + n_b)
        idx_a, idx_b = perm[:n_a], perm[n_a:]
        _, itc_pa = mne.time_frequency.tfr_multitaper(
            epochs_pooled[idx_a], freqs=freqs, n_cycles=n_cycles,
            time_bandwidth=time_bandwidth, return_itc=True,
            decim=decim, n_jobs=1, verbose=False,
        )
        _, itc_pb = mne.time_frequency.tfr_multitaper(
            epochs_pooled[idx_b], freqs=freqs, n_cycles=n_cycles,
            time_bandwidth=time_bandwidth, return_itc=True,
            decim=decim, n_jobs=1, verbose=False,
        )
        null_diffs.append(itc_pa.data - itc_pb.data)
        if (i + 1) % 100 == 0:
            print(f'    Permutation {i+1}/{n_permutations}')

    null_diffs = np.array(null_diffs)   # (n_perm, n_ch, n_freqs, n_times)

    # Z-score: (observed - null_mean) / null_std  per channel/freq/time point
    null_mean = null_diffs.mean(axis=0)
    null_std  = null_diffs.std(axis=0)
    null_std[null_std == 0] = np.nan    # avoid divide-by-zero

    z_scores = (observed - null_mean) / null_std  # (n_ch, n_freqs, n_times)
    z_avg    = z_scores.mean(axis=0)              # average across channels

    # Significance mask: |Z| > 1.96 (two-tailed, alpha=0.05)
    sig_mask = np.abs(z_avg) > 1.96

    return z_scores, z_avg, sig_mask

# ── PLOTTING ───────────────────────────────────────────────────────────────────

def average_over_channels(tfr_obj, tmin=None, tmax=None):
    """Average across channels, optionally crop time axis."""
    data  = tfr_obj.data.mean(axis=0)
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


def _crop_z(z_avg, tmin, tmax, times):
    """Crop a (n_freqs, n_times) Z-score array to the plot time window."""
    mask = (times >= tmin) & (times <= tmax)
    return z_avg[:, mask]

def plot_itc_differences(diff_ro, diff_rand,
                         sig_mask_ro=None, sig_mask_rand=None,
                         z_avg_ro=None, z_avg_rand=None,
                         region_name='',
                         tmin=TMIN_PLOT, tmax=TMAX_PLOT,
                         alpha_box=ALPHA_BOX, beta_box=BETA_BOX,
                         vlines=VLINES, output_file=None):
    """
    Two-panel figure for one region:
      Left  : ITC(Match) - ITC(Rule-Order)
      Right : ITC(Match) - ITC(Random)
    Black contours show p < 0.05 (uncorrected) where sig masks are provided.
    """
    # times_ro,   freqs_ro,   data_ro   = average_over_channels(diff_ro,   tmin, tmax)
    # times_rand, freqs_rand, data_rand = average_over_channels(diff_rand, tmin, tmax)

    # vmax = max(np.abs(data_ro).max(), np.abs(data_rand).max())
    # norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = 'RdBu_r'
    t_ms = 1000   # seconds -> ms

# Use Z-scores if available, otherwise fall back to raw ITC difference
    use_zscore = (z_avg_ro is not None) and (z_avg_rand is not None)

    if use_zscore:
        # z_avg is already (n_freqs, n_times) — just crop time
        times_ro,  freqs_ro,  _  = average_over_channels(diff_ro,   tmin, tmax)
        times_rand, freqs_rand, _ = average_over_channels(diff_rand, tmin, tmax)
        data_ro,   data_rand      = _crop_z(z_avg_ro,   tmin, tmax, diff_ro.times), \
                                    _crop_z(z_avg_rand, tmin, tmax, diff_rand.times)
        vmax     = max(np.abs(data_ro).max(), np.abs(data_rand).max())
        norm     = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cb_label = 'Z-score of ITC Difference'
    else:
        times_ro,  freqs_ro,  data_ro   = average_over_channels(diff_ro,   tmin, tmax)
        times_rand, freqs_rand, data_rand = average_over_channels(diff_rand, tmin, tmax)
        vmax     = max(np.abs(data_ro).max(), np.abs(data_rand).max())
        norm     = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cb_label = 'ITC Difference (Match - Other)'

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(
        1, 3, width_ratios=[1, 1, 0.05],
        left=0.07, right=0.93, bottom=0.12, top=0.88, wspace=0.28,
    )
    ax_ro   = fig.add_subplot(gs[0, 0])
    ax_rand = fig.add_subplot(gs[0, 1])
    ax_cb   = fig.add_subplot(gs[0, 2])

    panels = [
        (ax_ro,   times_ro,   freqs_ro,   data_ro,   sig_mask_ro,   'ITC  Match - Rule-Order'),
        (ax_rand, times_rand, freqs_rand, data_rand, sig_mask_rand, 'ITC  Match - Random'),
    ]

    for ax, times, freqs, data, sig_mask, title in panels:
        ax.pcolormesh(times * t_ms, freqs, data,
                      norm=norm, cmap=cmap, shading='auto')
        if sig_mask is not None:
            ax.contour(times * t_ms, freqs, sig_mask.astype(float),
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
        ax.set_ylim(freqs[0], freqs[-1])
        ax.set_xlabel('Time since Onset of Stimulus (ms)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    #cb_label = 'ITC Difference (Match - Other)'
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb
    ).set_label(cb_label, fontsize=9)

    title_str = 'Phase-Locked Coherence Difference'
    if region_name:
        title_str += f'  [{region_name.capitalize()}]'
    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=0.96)

    if output_file is not None:
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f'  Figure saved -> {output_file}')
    else:
        plt.show()
    return fig


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def run_pipeline():
    # 1. Load and label conditions
    print('Loading epochs and behavioural data...')
    cond_epochs = load_and_label(EPOCHS_FILE, BEH_FILE)

    # 2. Get full magnetometer channel list from match epochs
    all_ch = cond_epochs['match'].copy().pick('mag', exclude='bads').ch_names

    # 3. Map regions to channel lists and report counts
    region_channels = {}
    print('\nChannel counts per region:')
    for region, patterns in REGIONS.items():
        ch_list = channels_for_region(all_ch, patterns, ch_suffix=CH_SUFFIX)
        region_channels[region] = ch_list
        print(f'  {region:12s}: {len(ch_list)} channels  (e.g. {ch_list[:3]})')

    all_results = {}

    # 4. Run ITC pipeline independently for each region
    for region, ch_list in region_channels.items():
        sep = '=' * 50
        print(f'\n{sep}')
        print(f'  REGION: {region.upper()}  ({len(ch_list)} channels)')
        print(sep)

        if not ch_list:
            print(f'  No channels found for {region}, skipping.')
            continue

        # Channel selection per condition
        epochs_match = select_channels(cond_epochs['match'],      ch_list)
        epochs_ro    = select_channels(cond_epochs['rule_order'], ch_list)
        epochs_rand  = select_channels(cond_epochs['random'],     ch_list)

        # ITC per condition
        itc_match = compute_itc(epochs_match, FREQS, N_CYCLES, TIME_BANDWIDTH,
                                DECIM, f'match/{region}')
        itc_ro    = compute_itc(epochs_ro,    FREQS, N_CYCLES, TIME_BANDWIDTH,
                                DECIM, f'rule_order/{region}')
        itc_rand  = compute_itc(epochs_rand,  FREQS, N_CYCLES, TIME_BANDWIDTH,
                                DECIM, f'random/{region}')

        # Difference maps
        diff_ro   = itc_difference(itc_match, itc_ro,   f'Match-RuleOrder/{region}')
        diff_rand = itc_difference(itc_match, itc_rand, f'Match-Random/{region}')

        # Permutation tests (optional — set N_PERMUTATIONS=0 to skip)
        sig_mask_ro = sig_mask_rand = None
        if N_PERMUTATIONS > 0:
            print('  Permutation test: match vs rule_order...')
            _, z_avg_ro,   sig_mask_ro   = permutation_test_itc(
                epochs_match, epochs_ro, FREQS, N_CYCLES,
                TIME_BANDWIDTH, DECIM, N_PERMUTATIONS,
            )
            print('  Permutation test: match vs random...')
            _, z_avg_rand, sig_mask_rand = permutation_test_itc(
                epochs_match, epochs_rand, FREQS, N_CYCLES,
                TIME_BANDWIDTH, DECIM, N_PERMUTATIONS,
            )

        # Plot and save
        out_file = OUTPUT_DIR / f'itc_difference_{region}.png'
        fig = plot_itc_differences(
            diff_ro, diff_rand,
            sig_mask_ro=sig_mask_ro,
            sig_mask_rand=sig_mask_rand,
            z_avg_ro=z_avg_ro if N_PERMUTATIONS > 0 else None,      # ADD
            z_avg_rand=z_avg_rand if N_PERMUTATIONS > 0 else None,  # ADD
            region_name=region,
            output_file=out_file,
        )

        all_results[region] = dict(
            epochs_match=epochs_match, epochs_ro=epochs_ro, epochs_rand=epochs_rand,
            itc_match=itc_match, itc_ro=itc_ro, itc_rand=itc_rand,
            diff_ro=diff_ro, diff_rand=diff_rand,
            sig_mask_ro=sig_mask_ro, sig_mask_rand=sig_mask_rand,
            fig=fig,
        )
    
    print('\nDone. Output files:')
    for region in all_results:
        print(f'  {OUTPUT_DIR}/itc_difference_{region}.png')

    return all_results


if __name__ == '__main__':
    results = run_pipeline()