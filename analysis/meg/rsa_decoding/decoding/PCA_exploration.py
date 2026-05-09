import mne 
import torch
import numpy as np
import pandas as pd

EPOCHS_PATH = '/home/timmy/storage/NEU502B/brain-binding/tim/ported_results/sub-001_ses-01_task-binding_epo.fif'
epochs = mne.read_epochs(EPOCHS_PATH)
df_epochs_events = pd.DataFrame(epochs.metadata)
BEHAVIORAL_PATH = '/home/timmy/storage/NEU502B/brain-binding/data/2026-04-10/sub-001_events.csv'


