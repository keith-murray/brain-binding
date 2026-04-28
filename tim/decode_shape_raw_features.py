import sys, os

sys.path.insert(0, os.path.join(os.getcwd(), '..'))  # make toolkit importable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
mne.set_log_level('WARNING')

from toolkit import (
    RawAmplitudeFeature, TimeFrequencyFeature, RiemannianCovarianceFeature,
    RidgeLogisticDecoder, SVMDecoder,
    CVSplitter, cross_validate,
)

EPOCHS_PATH = '/home/timmy/storage/NEU502B/brain-binding/tim/ported_results/sub-001_ses-01_task-binding_epo.fif'
BEHAV_PATH  = '/home/timmy/storage/NEU502B/brain-binding/data/2026-04-10/sub-001_events.csv'

epochs = mne.read_epochs(EPOCHS_PATH, preload=True)
df = pd.read_csv(BEHAV_PATH).sort_values('trial').reset_index(drop=True)

sfreq = epochs.info['sfreq']
times = epochs.times
n_trials = len(df)
print(f'Epochs shape: {epochs.get_data(picks="meg").shape}')
print(f'Trials: {n_trials}, sfreq: {sfreq} Hz, times: {times[0]:.3f}–{times[-1]:.3f} s')

STAGE_OFFSETS = {
    'fixation': 0, 'rule1': 1, 'rule2': 2, 'rule3': 3,
    'transition': 4, 'test1': 5, 'test2': 6, 'test3': 7, 'response': 8,
}

def get_stage(stage_name: str):
    """Return (X, times) for the given within-trial stage across all trials."""
    offset = STAGE_OFFSETS[stage_name]
    idx = [9 * t + offset for t in range(n_trials)]
    X = epochs[idx].get_data(picks='meg').astype(np.float32)
    return X

# Binary label: ABA=0, ABB=1
y_B = (df['B_stim'] == 'star').values.astype(int)

