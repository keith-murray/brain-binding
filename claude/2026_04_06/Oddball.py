#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Passive Auditory Oddball - FieldTrip Style, EyeLink, with Blink Trials
========================================================================
Purpose: Pilot data collection for OPM MEG analysis with MNE
         (ERF averaging, time-frequency decomposition, source localization)

Paradigm Details (based on FieldTrip oddball tutorial):
    - Standard tone: 1000 Hz sine wave, 400 ms, 50 ms cosine ramp
    - Deviant tone:  1200 Hz sine wave, 400 ms, 50 ms cosine ramp
    - Deviant appears after every 3-7 standards (randomized)
    - ISI: 700-900 ms uniform jitter (offset to onset)
    - Passive listening with fixation cross
    - Blink trial (green fixation cross) after each deviant

Trigger Scheme (VPIXX Pixel Mode Blue):
    - 1x1 pixel patch, upper-left corner
    - Standard (trigger 1): RGB = (0, 0, 1)
    - Deviant  (trigger 2): RGB = (0, 0, 2)
    - Blink    (trigger 4): RGB = (0, 0, 4)
    - Reset    (trigger 0): RGB = (0, 0, 0)

Blink Trials:
    - Fixation cross turns green after each deviant trial
    - Subject blinks freely during this period
    - Keeps blink artifacts away from the MMN window (~150-250 ms)
    - No tone or trigger during blink trials

Eye Tracker:
    - SR Research EyeLink via pylink
    - Continuous recording throughout experiment
    - Event messages: standard_onset, deviant_onset, blink_cue
    - EDF file downloaded at end of session

Output:
    - CSV log: Trial_Num, Condition, onset_time, ISI
    - EDF file: eye tracking data with event messages
"""

import numpy as np
from psychopy import visual, sound, core, event, gui, data, monitors
from psychopy import logging
import pylink
import os
import sys
import csv
import platform
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

# Show only critical log messages in the PsychoPy console
logging.console.setLevel(logging.CRITICAL)

# Switch to the script folder
script_path = os.path.dirname(sys.argv[0])
if len(script_path) != 0:
    os.chdir(script_path)

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

# Tones
STANDARD_FREQ = 1000       # Hz
DEVIANT_FREQ = 1200        # Hz
TONE_DURATION = 0.400      # seconds
RAMP_DURATION = 0.050      # seconds (cosine rise/fall)
SAMPLE_RATE = 44100        # Hz

# Trial structure - gap-based (FieldTrip style)
# A deviant appears after every MIN_GAP to MAX_GAP standards
N_TOTAL = 1000             # total stimulus trials (~24 min with blinks)
N_INITIAL_STANDARDS = 10   # standards before first deviant
MIN_GAP = 3                # min standards between deviants
MAX_GAP = 7                # max standards between deviants

# Timing
ISI_MIN = 0.700            # seconds (offset to onset)
ISI_MAX = 0.900            # seconds

# Blink trials
BLINK_DURATION = 1.500     # seconds - green fixation cross after deviants
BLINK_COLOR = 'green'      # fixation cross color during blink cue

# Trigger pixel (VPIXX Pixel Mode Blue)
TRIGGER_SIZE = 1          # pixels
TRIGGER_STANDARD = [0, 0, 1]    # RGB 0-255
TRIGGER_DEVIANT = [0, 0, 2]     # RGB 0-255
TRIGGER_BLINK = [0, 0, 4]       # RGB 0-255
TRIGGER_RESET = [0, 0, 0]       # RGB 0-255

# EyeLink settings
DUMMY_MODE = False         # set True to test without tracker connected


# =============================================================================
# FUNCTIONS
# =============================================================================

def make_tone(frequency, duration, sample_rate, ramp_duration):
    """Generate a pure tone with cosine rise/fall ramp.

    Parameters
    ----------
    frequency : float
        Tone frequency in Hz.
    duration : float
        Total tone duration in seconds.
    sample_rate : int
        Audio sample rate in Hz.
    ramp_duration : float
        Duration of cosine rise/fall ramp in seconds.

    Returns
    -------
    tone : np.ndarray
        Mono audio waveform, values in [-1, 1].
    """
    n_samples = int(duration * sample_rate)
    n_ramp = int(ramp_duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Pure tone
    tone = np.sin(2.0 * np.pi * frequency * t)

    # Cosine ramp envelope
    ramp_up = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_ramp) / n_ramp))
    ramp_down = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_ramp) / n_ramp))
    tone[:n_ramp] *= ramp_up
    tone[-n_ramp:] *= ramp_down

    return tone.astype(np.float32)


def generate_trial_sequence(n_total, n_initial, min_gap, max_gap):
    """Generate trial sequence using gap-range approach.

    Each deviant appears after a random number of standards drawn
    uniformly from [min_gap, max_gap]. The sequence is built until
    n_total trials are reached.

    Parameters
    ----------
    n_total : int
        Total number of trials.
    n_initial : int
        Number of initial standard trials.
    min_gap : int
        Minimum standards before each deviant.
    max_gap : int
        Maximum standards before each deviant.

    Returns
    -------
    sequence : list of str
        List of 'standard' or 'deviant' for each trial.
    """
    rng = np.random.default_rng()
    sequence = ['standard'] * n_initial

    while len(sequence) < n_total:
        # Random number of standards before next deviant
        gap = rng.integers(min_gap, max_gap + 1)

        # Add standards
        for _ in range(gap):
            if len(sequence) >= n_total:
                break
            sequence.append('standard')

        # Add deviant (if room)
        if len(sequence) < n_total:
            sequence.append('deviant')

    # Trim to exact length
    sequence = sequence[:n_total]

    # Ensure the sequence does not end on a deviant
    if sequence[-1] == 'deviant':
        sequence[-1] = 'standard'

    return sequence


def rgb_255_to_psychopy(rgb):
    """Convert RGB [0-255] to PsychoPy color space [-1, 1].

    Parameters
    ----------
    rgb : list
        [R, G, B] values in 0-255 range.

    Returns
    -------
    list
        [R, G, B] values in -1 to 1 range.
    """
    return [(c / 127.5) - 1.0 for c in rgb]


def terminate_task(win, el_tracker, edf_file, session_folder,
                   local_edf_name):
    """Gracefully shut down: stop recording, download EDF, close window.

    Parameters
    ----------
    win : visual.Window
        PsychoPy window.
    el_tracker : pylink.EyeLink
        EyeLink tracker object.
    edf_file : str
        EDF filename on the Host PC.
    session_folder : str
        Local folder to save the EDF file.
    local_edf_name : str
        Full-length local EDF filename (without extension).
    """
    if el_tracker.isConnected():
        # Stop recording if active
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            pylink.pumpDelay(100)
            el_tracker.stopRecording()

        # Put tracker in offline mode
        el_tracker.setOfflineMode()
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the EDF file on the Host PC
        el_tracker.closeDataFile()

        # Download the EDF file
        msg = visual.TextStim(win,
                              text='EDF data is transferring from '
                                   'EyeLink Host PC...',
                              color='white', height=30)
        msg.draw()
        win.flip()

        local_edf = os.path.join(session_folder,
                                 local_edf_name + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
            print(f"EDF saved to: {local_edf}")
        except RuntimeError as error:
            print(f"ERROR downloading EDF: {error}")

        # Close the link
        el_tracker.close()

    win.close()
    core.quit()
    sys.exit()


# =============================================================================
# PARTICIPANT INFO DIALOG
# =============================================================================

exp_info = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=exp_info, title='Passive Auditory Oddball',
                       order=['participant', 'session'])
if not dlg.OK:
    core.quit()

exp_info['date'] = data.getDateStr()

# Output directories - CSV log and EDF go to the same session folder
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# EDF filename: short version for Host PC (max 8 chars),
# full version for local download
sub = exp_info['participant'].replace('-', '').replace('_', '')
ses = exp_info['session'].replace('-', '').replace('_', '')
edf_fname = f"s{sub}s{ses}"[:8]    # e.g., "s001s01" - truncate to 8 chars
local_edf_name = (f"sub-{exp_info['participant']}"
                  f"_ses-{exp_info['session']}_task-oddball")

# Session folder for all outputs (EDF + CSV)
session_folder = os.path.join(results_folder, local_edf_name)
os.makedirs(session_folder, exist_ok=True)

# CSV log filename (same folder as EDF)
filename = os.path.join(session_folder, local_edf_name)

# =============================================================================
# EYELINK: CONNECT AND CONFIGURE
# =============================================================================

# Step 1: Connect to the EyeLink Host PC
if DUMMY_MODE:
    el_tracker = pylink.EyeLink(None)
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print(f"ERROR: {error}")
        core.quit()
        sys.exit()

# Step 2: Open an EDF data file on the Host PC
edf_file = edf_fname + ".EDF"
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print(f"ERROR: {err}")
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# Add a header to the EDF file
preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

# Step 3: Configure the tracker
el_tracker.setOfflineMode()

# Get tracker version
eyelink_ver = 0
if not DUMMY_MODE:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split('.')[0])
    print(f"Running experiment on {vstr}, version {eyelink_ver}")

# File and Link data control
file_event_flags = ('LEFT,RIGHT,FIXATION,SACCADE,BLINK,'
                    'MESSAGE,BUTTON,INPUT')
link_event_flags = ('LEFT,RIGHT,FIXATION,SACCADE,BLINK,'
                    'BUTTON,FIXUPDATE,INPUT')
if eyelink_ver > 3:
    file_sample_flags = ('LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,'
                         'GAZERES,BUTTON,STATUS,INPUT')
    link_sample_flags = ('LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,'
                         'STATUS,INPUT')
else:
    file_sample_flags = ('LEFT,RIGHT,GAZE,HREF,RAW,AREA,'
                         'GAZERES,BUTTON,STATUS,INPUT')
    link_sample_flags = ('LEFT,RIGHT,GAZE,GAZERES,AREA,'
                         'STATUS,INPUT')

el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# Calibration type
el_tracker.sendCommand("calibration_type = HV9")

# =============================================================================
# SETUP WINDOW AND STIMULI
# =============================================================================

# Window - adjust size/monitor to your setup
win = visual.Window(
    fullscr=True,
    color=[0, 0, 0],       # black background
    colorSpace='rgb255',
    units='pix',
    allowGUI=False,
    screen=0               # change if projector is on a different display
)

# Get window dimensions
win_w, win_h = win.size
scn_width = int(win_w)
scn_height = int(win_h)

# Send display coordinates to the tracker
el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1,
                                                    scn_height - 1)
el_tracker.sendCommand(el_coords)
dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendMessage(dv_coords)

# =============================================================================
# EYELINK: CALIBRATION GRAPHICS
# =============================================================================

# Configure the graphics environment for calibration
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)

# Calibration colors: white target on black background
foreground_color = (1, 1, 1)
background_color = (-1, -1, -1)
genv.setCalibrationColors(foreground_color, background_color)

# Calibration target type
genv.setTargetType('circle')
genv.setTargetSize(24)

# Calibration sounds
genv.setCalibrationSounds('', '', '')

# Request Pylink to use our PsychoPy window for calibration
pylink.openGraphicsEx(genv)

# Fixation cross (white on black)
fixation = visual.ShapeStim(
    win,
    vertices=((0, -15), (0, 15), (0, 0), (-15, 0), (15, 0)),
    lineWidth=3,
    closeShape=False,
    lineColor='white',
    colorSpace='named'
)

# Blink cue fixation cross (green) - same shape, different color
fixation_blink = visual.ShapeStim(
    win,
    vertices=((0, -15), (0, 15), (0, 0), (-15, 0), (15, 0)),
    lineWidth=3,
    closeShape=False,
    lineColor=BLINK_COLOR,
    colorSpace='named'
)

# Trigger pixel - 1x1 pixel in upper-left corner
trigger_x = -win_w / 2.0 + TRIGGER_SIZE / 2.0
trigger_y = win_h / 2.0 - TRIGGER_SIZE / 2.0

trigger_patch = visual.Rect(
    win,
    width=TRIGGER_SIZE,
    height=TRIGGER_SIZE,
    pos=(trigger_x, trigger_y),
    lineWidth=0,
    fillColor=rgb_255_to_psychopy(TRIGGER_RESET),
    fillColorSpace='rgb',
    lineColor=rgb_255_to_psychopy(TRIGGER_RESET),
    lineColorSpace='rgb'
)

# =============================================================================
# GENERATE TONES
# =============================================================================

standard_wave = make_tone(STANDARD_FREQ, TONE_DURATION, SAMPLE_RATE,
                          RAMP_DURATION)
deviant_wave = make_tone(DEVIANT_FREQ, TONE_DURATION, SAMPLE_RATE,
                         RAMP_DURATION)

standard_sound = sound.Sound(value=standard_wave, sampleRate=SAMPLE_RATE,
                             stereo=False)
deviant_sound = sound.Sound(value=deviant_wave, sampleRate=SAMPLE_RATE,
                            stereo=False)

# =============================================================================
# GENERATE TRIAL SEQUENCE AND ISIs
# =============================================================================

trial_sequence = generate_trial_sequence(
    N_TOTAL, N_INITIAL_STANDARDS, MIN_GAP, MAX_GAP
)

n_deviants = trial_sequence.count('deviant')
n_standards = trial_sequence.count('standard')
est_stim_time = N_TOTAL * (TONE_DURATION + (ISI_MIN + ISI_MAX) / 2.0)
est_blink_time = n_deviants * BLINK_DURATION
est_duration = (est_stim_time + est_blink_time) / 60.0
print(f"Sequence: {n_standards} standards, {n_deviants} deviants "
      f"({n_deviants/N_TOTAL*100:.1f}%)")
print(f"Blink trials: {n_deviants} (after each deviant)")
print(f"Estimated run time: {est_duration:.1f} minutes")

# Pre-generate all ISIs
rng = np.random.default_rng()
isis = rng.uniform(ISI_MIN, ISI_MAX, size=N_TOTAL)

# =============================================================================
# EYELINK: CALIBRATE
# =============================================================================

# Show calibration instructions
cal_msg = visual.TextStim(
    win,
    text=(
        "Before we begin the experiment, we need to\n"
        "calibrate the eye tracker.\n\n"
        "A dot will appear at various locations on the screen.\n"
        "Simply follow the dot with your eyes. Do not anticipate\n"
        "where it will move - wait for it to appear in its new\n"
        "location before looking there.\n\n"
        "We may need to repeat the calibration, so please\n"
        "be patient.\n\n"
        "Experimenter: Press ENTER to begin calibration."
    ),
    color='white',
    height=30,
    wrapWidth=800
)
cal_msg.draw()
win.flip()
event.waitKeys(keyList=['return'])

# Run calibration (skip in Dummy Mode)
if not DUMMY_MODE:
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print(f"ERROR: {err}")
        el_tracker.exitCalibration()

# =============================================================================
# EYELINK: DRIFT CHECK & START RECORDING
# =============================================================================

# Drift check at fixation (center of screen)
while not DUMMY_MODE:
    if (not el_tracker.isConnected()) or el_tracker.breakPressed():
        terminate_task(win, el_tracker, edf_file, session_folder,
                       local_edf_name)
    try:
        error = el_tracker.doDriftCorrect(int(scn_width / 2.0),
                                           int(scn_height / 2.0), 1, 1)
        if error is not pylink.ESC_KEY:
            break
    except Exception:
        pass

# Put tracker in offline mode before starting recording
el_tracker.setOfflineMode()

# Start continuous recording for the entire experiment
try:
    el_tracker.startRecording(1, 1, 1, 1)
except RuntimeError as error:
    print(f"ERROR: {error}")
    terminate_task(win, el_tracker, edf_file, session_folder,
                   local_edf_name)

# Allow the tracker to cache some samples
pylink.pumpDelay(100)

# Mark the start of the experiment in the EDF
el_tracker.sendMessage('EXPERIMENT_START')

# =============================================================================
# PRE-EXPERIMENT INSTRUCTIONS
# =============================================================================

# Show task instructions before starting trials
instructions = visual.TextStim(
    win,
    text=(
        "You will hear a series of tones. Please ignore the\n"
        "sounds and keep your eyes on the fixation cross.\n\n"
        "When the fixation cross turns green, please blink.\n"
        "When it is white, try not to blink.\n\n"
        "Stay still and relaxed.\n\n"
        "Experimenter: Press SPACE to begin experiment."
    ),
    color='white',
    height=30,
    wrapWidth=800
)
instructions.draw()
win.flip()
event.waitKeys(keyList=['space'])

# Brief pause before starting
fixation.draw()
trigger_patch.fillColor = rgb_255_to_psychopy(TRIGGER_RESET)
trigger_patch.draw()
win.flip()
core.wait(2.0)

# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

clock = core.Clock()
trial_log = []

for trial_num in range(N_TOTAL):

    # Check for escape key
    if event.getKeys(keyList=['escape']):
        el_tracker.sendMessage('EXPERIMENT_ABORTED trial %d' % (trial_num + 1))
        break

    # Check tracker connection
    if not DUMMY_MODE:
        if (not el_tracker.isConnected()) or el_tracker.breakPressed():
            break

    # Determine condition
    condition = trial_sequence[trial_num]
    isi = isis[trial_num]

    # Select tone and trigger color
    if condition == 'standard':
        tone = standard_sound
        trig_color = rgb_255_to_psychopy(TRIGGER_STANDARD)
        el_msg = 'standard_onset'
    else:
        tone = deviant_sound
        trig_color = rgb_255_to_psychopy(TRIGGER_DEVIANT)
        el_msg = 'deviant_onset'

    # --- STIMULUS ONSET: draw trigger + fixation, play tone on same frame ---
    fixation.draw()
    trigger_patch.fillColor = trig_color
    trigger_patch.draw()

    # Mark trial in EDF (increments trial counter to match PsychoPy log)
    el_tracker.sendMessage('TRIALID %d' % (trial_num + 1))

    # Update Host PC status bar
    el_tracker.sendCommand(
        "record_status_message 'TRIAL %d/%d  %s'" %
        (trial_num + 1, N_TOTAL, condition)
    )

    # Schedule sound and EyeLink message to fire at the flip
    win.callOnFlip(tone.play)
    win.callOnFlip(el_tracker.sendMessage, el_msg)
    onset_time = win.flip()

    # Let tone play for its duration
    core.wait(TONE_DURATION)

    # --- TRIGGER RESET: return pixel to black during ISI ---
    fixation.draw()
    trigger_patch.fillColor = rgb_255_to_psychopy(TRIGGER_RESET)
    trigger_patch.draw()
    win.flip()

    # Log trial
    trial_log.append({
        'Trial_Num': trial_num + 1,
        'Condition': condition,
        'onset_time': f"{onset_time:.6f}",
        'ISI': f"{isi:.4f}"
    })

    # Wait for ISI (offset to next onset)
    core.wait(isi)

    # --- BLINK TRIAL: green fixation after each deviant ---
    if condition == 'deviant':
        # Show green fixation cross with blink trigger
        fixation_blink.draw()
        trigger_patch.fillColor = rgb_255_to_psychopy(TRIGGER_BLINK)
        trigger_patch.draw()

        el_tracker.sendMessage('blink_cue')
        el_tracker.sendCommand(
            "record_status_message 'TRIAL %d/%d  BLINK'" %
            (trial_num + 1, N_TOTAL)
        )
        blink_onset_time = win.flip()

        core.wait(BLINK_DURATION)

        # Return to white fixation, reset trigger
        fixation.draw()
        trigger_patch.fillColor = rgb_255_to_psychopy(TRIGGER_RESET)
        trigger_patch.draw()
        win.flip()

        # Log blink trial
        trial_log.append({
            'Trial_Num': trial_num + 1,
            'Condition': 'blink',
            'onset_time': f"{blink_onset_time:.6f}",
            'ISI': ''
        })

# =============================================================================
# STOP RECORDING
# =============================================================================

el_tracker.sendMessage('EXPERIMENT_END')

# Add 100 ms to catch final events
pylink.pumpDelay(100)
el_tracker.stopRecording()

# =============================================================================
# SAVE LOG FILE
# =============================================================================

csv_filename = filename + '.csv'
with open(csv_filename, 'w', newline='') as f:
    writer = csv.DictWriter(
        f, fieldnames=['Trial_Num', 'Condition', 'onset_time', 'ISI']
    )
    writer.writeheader()
    writer.writerows(trial_log)

print(f"Log saved to: {csv_filename}")

# =============================================================================
# END - DOWNLOAD EDF AND CLOSE
# =============================================================================

end_msg = visual.TextStim(
    win,
    text="The experiment is complete.\n\nThank you!\n\nPress SPACE to exit.",
    color='white',
    height=30,
    wrapWidth=800
)
end_msg.draw()
win.flip()
event.waitKeys(keyList=['space'])

terminate_task(win, el_tracker, edf_file, session_folder,
               local_edf_name)
