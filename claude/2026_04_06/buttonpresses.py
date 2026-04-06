# =============================================================================
# IMPORTS — at top of script
# =============================================================================

from pypixxlib import _libdpx as dp
from psychopy.data import ExperimentHandler, TrialHandler


# =============================================================================
# CONFIGURATION — in your parameters/info dict
#
# The three lists below must all be the same length and correspond index-for-index.
# Add or remove entries to match your response mapping.
# =============================================================================

info['buttonDINAssignment'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Full list of DIN bit positions, one per physical button on the box.
# Position in this list corresponds to the bit position in the hardware bitmask.

info['key_list'] = [0, 5]
# Subset of DIN bit positions to treat as valid responses.
# Only presses at these positions will be returned by get_response().
# Example: [0, 5] for left and right index fingers.

info['corr_key_list'] = ['z', 'm']
# Label for each entry in key_list, used for correctness checking.
# Can be any strings that match your trial data's corrAns field.
# Example: ['z', 'm'], ['left', 'right'], ['1', '2'], etc.
# Must be the same length and order as key_list.

info['pressedValue'] = '1'
# The bit value that indicates a button is pressed.
# '1' for standard MRI and MEG compatible boxes;

info['exitButton'] = 'escape'
# Keyboard key that aborts the experiment. Checked alongside button box events.


# =============================================================================
# VPIXX HARDWARE INIT — before opening the window
# =============================================================================

dp.DPxOpen()
dp.DPxEnableDoutPixelMode()   # enable pixel-mode triggers on the output port
dp.DPxStopAllScheds()
dp.DPxUpdateRegCache()


# =============================================================================
# DIN LOG HELPER — define once, call throughout
# =============================================================================

def responsepixx_status(info, status='start'):
    """Start, pause, or resume the DIN button log.

    Call with status='start' once at experiment start.
    Call with status='resume' to open the response window before each trial.
    Call with status='pause' to close the response window after each response,
    preventing button presses from bleeding into the next trial.
    """

    if status == 'start':
        # Allocate the log buffer on the hardware and begin logging.
        myLog = dp.DPxSetDinLog(12e6, 500)  # 12M sample buffer, 500 log frames
        dp.DPxStartDinLog()
        dp.DPxUpdateRegCache()
        info['myLog'] = myLog               # store handle for later reads

    elif status == 'pause':
        dp.DPxStopDinLog()
        dp.DPxUpdateRegCache()

    elif status == 'resume':
        dp.DPxStartDinLog()
        dp.DPxUpdateRegCache()

    return info


# =============================================================================
# RESPONSE READING — define once, call every frame during response window
# =============================================================================

def get_response(info):
    """Poll the DIN log for a button press.

    Returns
    -------
    resp : str or None
        The label from corr_key_list corresponding to the pressed button,
        or a keyboard key string (e.g. 'escape') if a keyboard key was pressed.
        None if no press was detected this frame.
    resp_time : float or None
        Hardware timestamp of the press in seconds (VPixx clock), or None.
    """

    resp      = None
    resp_time = None

    # Sync local register cache with hardware state.
    dp.DPxUpdateRegCache()

    # Check how many new log frames have arrived since last read.
    dp.DPxGetDinStatus(info['myLog'])
    newEvents = info['myLog']['newLogFrames']

    if newEvents > 0:
        # Read new events. Each event is [timestamp_seconds, din_state_bitmask].
        eventList = dp.DPxReadDinLog(info['myLog'], newEvents)

        for x in eventList:
            timestamp = x[0]   # hardware clock time in seconds
            din_state = x[1]   # integer bitmask — each bit = one button

            # Convert bitmask to binary string and take the last 10 bits
            # (ResponsePixx buttons occupy bits 0–9).
            binary_str   = bin(din_state)[2:]
            last_10_bits = binary_str[-10:].zfill(10)

            # Reverse so that list index matches button bit position
            # (hardware counts bits right→left; Python indexes left→right).
            last_10_bits = last_10_bits[::-1]

            # Find all bit positions that are high (button pressed).
            high_bit_positions = [i for i, bit in enumerate(last_10_bits)
                                  if bit == info['pressedValue']]

            # Only accept a clean single-button press (ignore chords).
            if len(high_bit_positions) == 1:
                pos          = high_bit_positions[0]
                button_index = info['buttonDINAssignment'][pos]

                if button_index in info['key_list']:
                    # Convert button index → response label via corr_key_list.
                    idx       = info['key_list'].index(button_index)
                    resp      = info['corr_key_list'][idx]
                    resp_time = timestamp
                    return resp, resp_time   # return immediately on first valid press

    # Also check the keyboard (escape / experimenter keys).
    keys = getKeys(keyList=[info['exitButton']], timeStamped=True)
    if keys:
        resp, _ = keys[0]

    return resp, resp_time


# =============================================================================
# DATA RECORDING HELPER — define once, call once per trial
# =============================================================================

def record_data(trials, exp, resp, rt, corr, **extra_fields):
    """Write trial data to the PsychoPy CSV log.

    Parameters
    ----------
    trials : TrialHandler
        The active trial handler for the current block.
    exp : ExperimentHandler
        The experiment-level handler that owns the output file.
    resp : str or None
        The response label returned by get_response().
    rt : float or None
        Reaction time in seconds (resp_time - startTime).
    corr : int or float
        Correctness: 1 = correct, 0 = incorrect, float('nan') = no response.
    **extra_fields : dict
        Any additional trial-level fields you want logged, e.g.:
        record_data(..., block='run1', condition='match', onset_time=t)
    """

    # Core response fields
    trials.addData('resp', resp)
    trials.addData('rt',   rt)
    trials.addData('corr', corr)

    # Any extra fields passed in as keyword arguments
    for field, value in extra_fields.items():
        trials.addData(field, value)

    # Advance to the next row in the CSV — must be called exactly once per trial.
    # ExperimentHandler writes incrementally, so data is saved even if the
    # experiment crashes before the end of the session.
    exp.nextEntry()


# =============================================================================
# EXPERIMENT AND DATA HANDLER INIT — after window opens, before trial loop
# =============================================================================

# ExperimentHandler owns the output CSV file.
exp = ExperimentHandler(
    name='my_experiment',
    extraInfo=info,
    dataFileName=os.path.join(info['output_dir'], info['filename'])
)

# TrialHandler manages trial order for one block.
# Replace trialList and nReps to match your design.
trials = TrialHandler(
    trialList=my_trial_list,   # list of dicts, one per trial
    nReps=1,
    method='random',
    name='main_block'
)
exp.addLoop(trials)   # register the block with the experiment handler

# Start the DIN log
info = responsepixx_status(info, status='start')
info = responsepixx_status(info, status='pause')   # pause immediately; resume per trial


# =============================================================================
# TRIAL LOOP
# =============================================================================

for trial in trials:

    # --- Open the response window ---
    info      = responsepixx_status(info, status='resume')
    startTime = dp.DPxGetTime()   # hardware timestamp at stimulus onset

    resp      = None
    resp_time = None
    rt        = None

    # --- Frame loop: draw stimuli and poll for response each frame ---
    while True:

        # ... draw your stimuli here ...
        win.flip()

        resp, resp_time = get_response(info)

        if resp is not None:
            rt = resp_time - startTime   # both from the VPixx hardware clock
            break

        # Optional: enforce a response deadline
        if dp.DPxGetTime() - startTime > RESPONSE_DEADLINE:
            break

    # --- Close the response window ---
    info = responsepixx_status(info, status='pause')

    # --- Score correctness ---
    # trial['corrAns'] should be a label matching one of your corr_key_list values.
    if resp is None:
        corr = float('nan')
        rt   = float('nan')
    elif resp == info['exitButton']:
        core.quit()
    elif resp == trial['corrAns']:
        corr = 1
    else:
        corr = 0

    # --- Write to CSV ---
    # Add any experiment-specific fields as keyword arguments.
    record_data(
        trials, exp,
        resp=resp,
        rt=rt,
        corr=corr,
        onset_time=startTime,        # example extra field
        condition=trial['condition'] # example extra field
    )


# =============================================================================
# EXPERIMENT END — before closing the window
# =============================================================================

dp.DPxDisableDoutPixelMode()
dp.DPxWriteRegCache()
dp.DPxClose()