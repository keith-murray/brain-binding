#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Relational Reasoning Task — MEG Version
========================================
Purpose: MEG data collection for relational reasoning analysis.

Paradigm:
    - Rule phase: 3 shapes shown sequentially following ABA or ABB pattern
    - Test phase: 3 shapes shown sequentially following ABA or ABB pattern
    - Subject presses 1 if test matches the rule, 2 if it does not
    - 50% match trials; mismatch trials are 50/50:
        - rule_order: same test shapes, opposite rule (ABA→ABB or ABB→ABA)
        - random: 3rd test shape is randomly drawn from shapes not in the test pair

Trigger Scheme (VPIXX Pixel Mode Blue, powers of 2):
    - Reset       (  0): RGB = (0, 0, 0)   — between events
    - Fixation    (  1): RGB = (0, 0, 1)   — trial-start white cross
    - Rule stim   (  2): RGB = (0, 0, 2)   — any of the 3 rule stimuli
    - Test stim   (  4): RGB = (0, 0, 4)   — any of the 3 test stimuli
    - Transition  (  8): RGB = (0, 0, 8)   — yellow cross (rule→test gap)
    - Correct     ( 16): RGB = (0, 0, 16)  — green cross (correct feedback)
    - Incorrect   ( 32): RGB = (0, 0, 32)  — red cross (incorrect feedback)
    - Blink       ( 64): RGB = (0, 0, 64)  — blue cross (blink cue)

Blink Trials:
    - Blue fixation cross after each trial's feedback
    - Subject blinks freely during this period
    - Keeps blink artifacts outside the analysis window

Button Box (VPixx ResponsePixx via pypixxlib):
    - DIN bit 0 → key '1' (match)
    - DIN bit 1 → key '2' (no-match)
    - Keyboard 'escape' aborts the session

Output:
    - CSV log per session (manual, BIDS-style naming)
    - PsychoPy ExperimentHandler CSV (instructor-recommended logging)
    - PsychoPy log file
"""

import csv
import random
from datetime import datetime
from itertools import permutations
from pathlib import Path

import numpy as np
from psychopy import core, data, event, gui, logging, visual
from psychopy.data import ExperimentHandler, TrialHandler

# Try importing pypixxlib; fall back to keyboard-only dummy mode if unavailable
try:
    from pypixxlib import _libdpx as dp
    DUMMY_MODE = True
except ImportError:
    DUMMY_MODE = True

logging.console.setLevel(logging.CRITICAL)


# =============================================================================
# GRACEFUL EXIT
# =============================================================================

class GracefulExit(Exception):
    """Raised when the experimenter presses Escape to abort the session."""


def interruptible_wait(duration: float) -> None:
    """Wait for *duration* seconds; raise GracefulExit if Escape is pressed."""
    clock = core.Clock()
    while clock.getTime() < duration:
        if event.getKeys(keyList=['escape']):
            raise GracefulExit()
        core.wait(0.005, hogCPUperiod=0)


# =============================================================================
# CONSTANTS
# =============================================================================

STIM_NAMES = ['circle', 'rectangle', 'star', 'triangle']
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Timing (seconds)
FIXATION_DURATION   = 0.500
STIM_DURATION       = 0.500
FEEDBACK_DURATION   = 0.500
BLINK_DURATION      = 1.500
RESPONSE_DEADLINE   = 2.000
TRANSITION_DURATION = 1.500   # fixed gap between rule and test phases

ISI_MEAN = 1.0
ISI_SD   = 0.2
ISI_MIN  = 0.0

ITI_BASE = 3.0
ITI_SD   = 1.0
ITI_MIN  = 0.0

# Task structure
N_RUNS         = 4
TRIALS_PER_RUN = 30   # 4 × 30 = 120 total
RULE_REPS      = 5    # 24 configs × 5 = 120

# Triggers (VPIXX Pixel Mode Blue channel, powers of 2)
TRIGGER_SIZE       = 1
TRIGGER_RESET      = [0, 0,   0]
TRIGGER_FIXATION   = [0, 0,   1]
TRIGGER_RULE       = [0, 0,   2]   # all 3 rule stims share this code
TRIGGER_TEST       = [0, 0,   4]   # all 3 test stims share this code
TRIGGER_TRANSITION = [0, 0,   8]   # yellow fixation (rule→test boundary)
TRIGGER_CORRECT    = [0, 0,  16]   # green fixation (correct feedback)
TRIGGER_INCORRECT  = [0, 0,  32]   # red fixation (incorrect feedback)
TRIGGER_BLINK      = [0, 0,  64]   # blue fixation (blink cue)

# Fixation cross colors
COLOR_FIXATION   = 'black'
COLOR_CORRECT    = 'green'
COLOR_INCORRECT  = 'red'
COLOR_BLINK      = 'blue'
COLOR_TRANSITION = 'yellow'

# Button box configuration (VPixx ResponsePixx DIN bits)
BUTTON_DIN_ASSIGNMENT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
BUTTON_KEY_LIST       = [0, 1]          # DIN bit positions for the two response buttons
BUTTON_LABEL_LIST     = ['1', '2']      # '1' = match, '2' = no-match
BUTTON_PRESSED_VALUE  = '1'

FIELDNAMES = [
    'block', 'sid', 'trial',
    'rule_type', 'A_stim', 'B_stim', 'rule_sequence',
    'test_type', 'X_stim', 'Y_stim', 'test_sequence',
    'match', 'mismatch_type',
    'response_key', 'correct', 'rt',
    'isi1', 'isi2', 'isi3', 'isi4', 'isi5', 'iti',
    't_fixation',
    't_rule1', 't_rule2', 't_rule3',
    't_test1', 't_test2', 't_test3',
    't_response',
]


# =============================================================================
# HELPERS
# =============================================================================

def rgb_255_to_psychopy(rgb: list) -> list:
    """Convert RGB [0–255] to PsychoPy color space [–1, 1]."""
    return [(c / 127.5) - 1.0 for c in rgb]


def jitter(mean: float, sd: float, min_val: float) -> float:
    return float(max(min_val, mean + np.random.default_rng().normal(0, sd)))


def make_fixation(win: visual.Window, color: str) -> visual.ShapeStim:
    return visual.ShapeStim(
        win,
        vertices=((0, -15), (0, 15), (0, 0), (-15, 0), (15, 0)),
        lineWidth=5,
        closeShape=False,
        lineColor=color,
        colorSpace='named',
    )


# =============================================================================
# TRIAL GENERATION
# =============================================================================

def gen_main_trials(seed: int) -> list:
    """Generate 120 trials with balanced match/mismatch and mismatch types.

    - 120 trials: 24 (A, B, rule_type) configs × 5 reps
    - Exactly 60 match, 60 mismatch
    - Among mismatch: exactly 30 rule_order, 30 random

    Match:
        Test sequence follows the same rule as the rule sequence,
        using new shapes X ≠ Y.

    Mismatch — rule_order:
        Same X, Y shapes but the opposite rule pattern
        (ABA rule → ABB test, or ABB rule → ABA test).

    Mismatch — random:
        Test sequence is [X, Y, Z] where Z is randomly chosen
        from the two shapes not in {X, Y}.
    """
    rng = random.Random(seed)

    configs = [
        {'A_stim': a, 'B_stim': b, 'rule_type': rule}
        for a, b in permutations(STIM_NAMES, 2)
        for rule in ['ABA', 'ABB']
    ]
    base = configs * RULE_REPS
    rng.shuffle(base)

    match_labels = [True] * 60 + [False] * 60
    rng.shuffle(match_labels)

    mismatch_types = ['rule_order'] * 30 + ['random'] * 30
    rng.shuffle(mismatch_types)
    mismatch_iter = iter(mismatch_types)

    trials = []
    for cfg, is_match in zip(base, match_labels):
        a, b, rule_type = cfg['A_stim'], cfg['B_stim'], cfg['rule_type']
        rule_seq = [a, b, a] if rule_type == 'ABA' else [a, b, b]

        x, y = rng.sample(STIM_NAMES, 2)

        if is_match:
            test_type = rule_type
            test_seq = [x, y, x] if rule_type == 'ABA' else [x, y, y]
            mismatch_type = ''
        else:
            mt = next(mismatch_iter)
            if mt == 'rule_order':
                test_type = 'ABB' if rule_type == 'ABA' else 'ABA'
                test_seq = [x, y, x] if test_type == 'ABA' else [x, y, y]
            else:
                other = [s for s in STIM_NAMES if s not in {x, y}]
                z = rng.choice(other)
                test_type = 'other'
                test_seq = [x, y, z]
            mismatch_type = mt

        trials.append({
            'rule_type': rule_type,
            'A_stim': a,
            'B_stim': b,
            'rule_sequence': ','.join(rule_seq),
            'test_type': test_type,
            'X_stim': x,
            'Y_stim': y,
            'test_sequence': ','.join(test_seq),
            'match': int(is_match),
            'mismatch_type': mismatch_type,
        })

    return trials


# =============================================================================
# VPIXX BUTTON BOX
# =============================================================================

def responsepixx_status(info: dict, status: str = 'start') -> dict:
    """Start, pause, or resume the VPixx DIN button log.

    Call with status='start' once at experiment start.
    Call with status='resume' before each response window.
    Call with status='pause' after each response window.
    """
    if DUMMY_MODE:
        return info

    if status == 'start':
        myLog = dp.DPxSetDinLog(12e6, 500)
        dp.DPxStartDinLog()
        dp.DPxUpdateRegCache()
        info['myLog'] = myLog
    elif status == 'pause':
        dp.DPxStopDinLog()
        dp.DPxUpdateRegCache()
    elif status == 'resume':
        dp.DPxStartDinLog()
        dp.DPxUpdateRegCache()

    return info


def get_response(info: dict) -> tuple:
    """Poll the DIN log for a button press.

    Returns
    -------
    resp : str or None
        Button label ('1' or '2') or 'escape'. None if no press this frame.
    resp_time : float or None
        VPixx hardware timestamp in seconds, or None.
    """
    resp      = None
    resp_time = None

    if not DUMMY_MODE:
        dp.DPxUpdateRegCache()
        dp.DPxGetDinStatus(info['myLog'])
        new_events = info['myLog']['newLogFrames']

        if new_events > 0:
            event_list = dp.DPxReadDinLog(info['myLog'], new_events)

            for x in event_list:
                timestamp = x[0]
                din_state = x[1]

                binary_str   = bin(din_state)[2:]
                last_10_bits = binary_str[-10:].zfill(10)[::-1]

                high_bits = [i for i, bit in enumerate(last_10_bits)
                             if bit == info['pressedValue']]

                if len(high_bits) == 1:
                    pos = high_bits[0]
                    button_index = info['buttonDINAssignment'][pos]

                    if button_index in info['key_list']:
                        idx       = info['key_list'].index(button_index)
                        resp      = info['corr_key_list'][idx]
                        resp_time = timestamp
                        return resp, resp_time

    # Keyboard fallback (escape abort, and keyboard responses in dummy mode)
    keys = event.getKeys(
        keyList=['escape'] + (BUTTON_LABEL_LIST if DUMMY_MODE else []),
        timeStamped=True,
    )
    if keys:
        resp = keys[0][0]

    return resp, resp_time


# =============================================================================
# EXPERIMENT HANDLER LOGGING
# =============================================================================

def record_data(trials_handler: TrialHandler, exp_handler: ExperimentHandler,
                resp, rt, corr, **extra_fields) -> None:
    """Write trial data to the PsychoPy ExperimentHandler CSV.

    Parameters
    ----------
    trials_handler : TrialHandler
        The active trial handler for the current run block.
    exp_handler : ExperimentHandler
        The experiment-level handler that owns the output file.
    resp : str or None
        Response label ('1', '2', or None for timeout).
    rt : float or None
        Reaction time in seconds.
    corr : int or float
        Correctness: 1 = correct, 0 = incorrect, float('nan') = no response.
    **extra_fields
        Any additional trial-level fields to log.
    """
    trials_handler.addData('resp', resp)
    trials_handler.addData('rt',   rt)
    trials_handler.addData('corr', corr)

    for field, value in extra_fields.items():
        trials_handler.addData(field, value)

    exp_handler.nextEntry()


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def _send_trigger(trigger_patch: visual.Rect, color: list) -> None:
    trigger_patch.fillColor = rgb_255_to_psychopy(color)


def show_fixation(win: visual.Window, fixation: visual.ShapeStim,
                  trigger_patch: visual.Rect) -> float:
    """Draw fixation with trigger 1; return onset time."""
    fixation.draw()
    _send_trigger(trigger_patch, TRIGGER_FIXATION)
    trigger_patch.draw()
    t = win.flip()
    core.wait(FIXATION_DURATION)
    return t


def show_stimulus(win: visual.Window, stim: visual.ImageStim,
                  stim_pos: tuple, trigger_color: list,
                  trigger_patch: visual.Rect) -> float:
    """Draw image stimulus with trigger; return onset time."""
    stim.pos = stim_pos
    stim.draw()
    _send_trigger(trigger_patch, trigger_color)
    trigger_patch.draw()
    t = win.flip()
    core.wait(STIM_DURATION)
    return t


def show_blank(win: visual.Window, fixation: visual.ShapeStim,
               trigger_patch: visual.Rect, duration: float,
               trigger_color: list = None) -> None:
    """Show fixation with reset trigger (or custom trigger) for given duration."""
    if trigger_color is None:
        trigger_color = TRIGGER_RESET
    fixation.draw()
    _send_trigger(trigger_patch, trigger_color)
    trigger_patch.draw()
    win.flip()
    if duration > 0:
        interruptible_wait(duration)


def show_response_window(win: visual.Window, fixation: visual.ShapeStim,
                         trigger_patch: visual.Rect,
                         info: dict) -> tuple:
    """Show fixation and poll for response.

    In hardware mode, polls the VPixx DIN log each frame.
    In dummy mode, falls back to keyboard.

    Returns (t_onset, response_key, rt).
    """
    info = responsepixx_status(info, status='resume')

    fixation.draw()
    _send_trigger(trigger_patch, TRIGGER_RESET)
    trigger_patch.draw()
    t_onset = win.flip()

    if not DUMMY_MODE:
        start_time = dp.DPxGetTime()

    psychopy_clock = core.Clock()
    response_key = None
    rt = None

    while True:
        fixation.draw()
        trigger_patch.draw()
        win.flip()

        resp, resp_time = get_response(info)

        if resp == 'escape':
            info = responsepixx_status(info, status='pause')
            raise GracefulExit()

        if resp is not None:
            if DUMMY_MODE:
                rt = psychopy_clock.getTime()
            else:
                rt = resp_time - start_time
            response_key = resp
            break

        elapsed = psychopy_clock.getTime()
        if elapsed >= RESPONSE_DEADLINE:
            break

    info = responsepixx_status(info, status='pause')
    return t_onset, response_key, rt


def show_feedback(win: visual.Window,
                  fixation_correct: visual.ShapeStim,
                  fixation_incorrect: visual.ShapeStim,
                  trigger_patch: visual.Rect,
                  correct: bool) -> None:
    """Briefly show green (correct) or red (incorrect) fixation cross."""
    fix     = fixation_correct   if correct else fixation_incorrect
    trigger = TRIGGER_CORRECT    if correct else TRIGGER_INCORRECT
    fix.draw()
    _send_trigger(trigger_patch, trigger)
    trigger_patch.draw()
    win.flip()
    core.wait(FEEDBACK_DURATION)


def show_blink_window(win: visual.Window, fixation_blink: visual.ShapeStim,
                      trigger_patch: visual.Rect) -> None:
    """Show blue fixation cross with blink trigger."""
    fixation_blink.draw()
    _send_trigger(trigger_patch, TRIGGER_BLINK)
    trigger_patch.draw()
    win.flip()
    core.wait(BLINK_DURATION)


def show_text_screen(win: visual.Window, text: str, info: dict,
                     trigger_patch: visual.Rect) -> None:
    """Show a text message and wait for any button box press."""
    msg = visual.TextStim(
        win=win,
        text=text,
        color='black',
        height=30,
        wrapWidth=win.size[0] * 0.8,
    )
    msg.draw()
    _send_trigger(trigger_patch, TRIGGER_RESET)
    trigger_patch.draw()
    win.flip()

    info = responsepixx_status(info, status='resume')
    while True:
        resp, _ = get_response(info)
        if resp == 'escape':
            info = responsepixx_status(info, status='pause')
            raise GracefulExit()
        if resp is not None:
            break
        core.wait(0.005, hogCPUperiod=0)
    info = responsepixx_status(info, status='pause')


# =============================================================================
# TASK RUNNER
# =============================================================================

def run_main_task(win, stimuli, stim_pos,
                  fixation, fixation_correct, fixation_incorrect,
                  fixation_blink, fixation_transition,
                  trigger_patch, manual_writer, manual_csv,
                  exp_handler, main_clock, info, sid, seed):

    trials = gen_main_trials(seed)

    for run_idx in range(N_RUNS):
        if run_idx > 0:
            show_text_screen(
                win,
                f'Break — run {run_idx} of {N_RUNS} complete.\n\n'
                'Rest for a moment, then press any button to continue.',
                info,
                trigger_patch,
            )

        run_trials = trials[run_idx * TRIALS_PER_RUN:(run_idx + 1) * TRIALS_PER_RUN]

        # Register this run's trials with the ExperimentHandler
        trials_handler = TrialHandler(
            trialList=run_trials,
            nReps=1,
            method='sequential',
            name=f'run{run_idx + 1}',
        )
        exp_handler.addLoop(trials_handler)

        for t_rel, trial in enumerate(trials_handler):
            t_idx = run_idx * TRIALS_PER_RUN + t_rel

            if event.getKeys(keyList=['escape']):
                raise GracefulExit()

            rule_seq = trial['rule_sequence'].split(',')
            test_seq = trial['test_sequence'].split(',')

            isi1    = jitter(ISI_MEAN, ISI_SD, ISI_MIN)
            isi2    = jitter(ISI_MEAN, ISI_SD, ISI_MIN)
            isi3    = TRANSITION_DURATION   # fixed; yellow fixation marks rule→test boundary
            isi4    = jitter(ISI_MEAN, ISI_SD, ISI_MIN)
            isi5    = jitter(ISI_MEAN, ISI_SD, ISI_MIN)
            iti_dur = jitter(ITI_BASE, ITI_SD, ITI_MIN)

            # 1. Trial-start fixation
            t_fixation = show_fixation(win, fixation, trigger_patch)

            # 2. Rule phase
            t_rule1 = show_stimulus(win, stimuli[rule_seq[0]], stim_pos,
                                    TRIGGER_RULE, trigger_patch)
            show_blank(win, fixation, trigger_patch, isi1)

            t_rule2 = show_stimulus(win, stimuli[rule_seq[1]], stim_pos,
                                    TRIGGER_RULE, trigger_patch)
            show_blank(win, fixation, trigger_patch, isi2)

            t_rule3 = show_stimulus(win, stimuli[rule_seq[2]], stim_pos,
                                    TRIGGER_RULE, trigger_patch)

            # 3. Transition cue (yellow fixation, fixed 1.5 s)
            show_blank(win, fixation_transition, trigger_patch, isi3,
                       trigger_color=TRIGGER_TRANSITION)

            # 4. Test phase
            t_test1 = show_stimulus(win, stimuli[test_seq[0]], stim_pos,
                                    TRIGGER_TEST, trigger_patch)
            show_blank(win, fixation, trigger_patch, isi4)

            t_test2 = show_stimulus(win, stimuli[test_seq[1]], stim_pos,
                                    TRIGGER_TEST, trigger_patch)
            show_blank(win, fixation, trigger_patch, isi5)

            t_test3 = show_stimulus(win, stimuli[test_seq[2]], stim_pos,
                                    TRIGGER_TEST, trigger_patch)

            # 5. Response window
            t_response, response_key, rt = show_response_window(
                win, fixation, trigger_patch, info
            )

            if response_key is not None and response_key != 'escape':
                correct = int((response_key == '1') == bool(trial['match']))
                corr_log = correct
            else:
                correct  = 0
                corr_log = float('nan')

            # 6. Feedback
            show_feedback(win, fixation_correct, fixation_incorrect,
                          trigger_patch, bool(correct))

            # 7. Blink window
            show_blink_window(win, fixation_blink, trigger_patch)

            # 8. ITI
            show_blank(win, fixation, trigger_patch, iti_dur)

            # --- Manual CSV log ---
            row = {fn: '' for fn in FIELDNAMES}
            row.update({
                'block':         f'run{run_idx + 1}',
                'sid':           sid,
                'trial':         t_idx,
                'rule_type':     trial['rule_type'],
                'A_stim':        trial['A_stim'],
                'B_stim':        trial['B_stim'],
                'rule_sequence': trial['rule_sequence'],
                'test_type':     trial['test_type'],
                'X_stim':        trial['X_stim'],
                'Y_stim':        trial['Y_stim'],
                'test_sequence': trial['test_sequence'],
                'match':         trial['match'],
                'mismatch_type': trial['mismatch_type'],
                'response_key':  response_key,
                'correct':       correct,
                'rt':            rt,
                'isi1': isi1, 'isi2': isi2, 'isi3': isi3,
                'isi4': isi4, 'isi5': isi5, 'iti': iti_dur,
                't_fixation': t_fixation,
                't_rule1': t_rule1, 't_rule2': t_rule2, 't_rule3': t_rule3,
                't_test1': t_test1, 't_test2': t_test2, 't_test3': t_test3,
                't_response': t_response,
            })
            manual_writer.writerow(row)
            manual_csv.flush()

            # --- PsychoPy ExperimentHandler log ---
            record_data(
                trials_handler, exp_handler,
                resp=response_key,
                rt=rt,
                corr=corr_log,
                block=f'run{run_idx + 1}',
                trial=t_idx,
                rule_type=trial['rule_type'],
                test_type=trial['test_type'],
                match=trial['match'],
                mismatch_type=trial['mismatch_type'],
                t_response=t_response,
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    exp_info = {'participant': '', 'session': '001'}
    dlg = gui.DlgFromDict(dictionary=exp_info, title='Relational Reasoning — MEG',
                          order=['participant', 'session'])
    if not dlg.OK:
        core.quit()

    exp_info['date'] = data.getDateStr()
    seed = int(datetime.now().timestamp() * 1000) % (2 ** 31)
    sid  = exp_info['participant']

    session_name = (f"sub-{sid}"
                    f"_ses-{exp_info['session']}"
                    f"_task-relational")
    data_dir = Path(__file__).parent.parent / 'data' / session_name
    data_dir.mkdir(parents=True, exist_ok=True)

    main_clock = core.Clock()
    logging.setDefaultClock(main_clock)
    log_path = data_dir / f'{session_name}.log'
    logging.LogFile(str(log_path), level=logging.INFO, filemode='w')
    logging.info(f'sid={sid}, seed={seed}, dummy_mode={DUMMY_MODE}')

    # --- VPixx hardware init (before opening window) ---
    if not DUMMY_MODE:
        dp.DPxOpen()
        dp.DPxEnableDoutPixelMode()
        dp.DPxStopAllScheds()
        dp.DPxUpdateRegCache()

    # --- Button box info dict ---
    info = {
        'buttonDINAssignment': BUTTON_DIN_ASSIGNMENT,
        'key_list':            BUTTON_KEY_LIST,
        'corr_key_list':       BUTTON_LABEL_LIST,
        'pressedValue':        BUTTON_PRESSED_VALUE,
        'exitButton':          'escape',
    }

    win = visual.Window(
        fullscr=True,
        color='white',
        colorSpace='rgb',
        units='pix',
        allowGUI=False,
        screen=0,   # change if projector is on a different display
    )
    win_w, win_h = win.size

    stim_size = (int(win_w * 0.14), int(win_h * 0.185))
    stim_pos  = (0, 0)

    stimuli = {
        name: visual.ImageStim(
            win=win,
            image=str(ASSETS_DIR / f'{name}.png'),
            size=stim_size,
        )
        for name in STIM_NAMES
    }

    fixation            = make_fixation(win, COLOR_FIXATION)
    fixation_correct    = make_fixation(win, COLOR_CORRECT)
    fixation_incorrect  = make_fixation(win, COLOR_INCORRECT)
    fixation_blink      = make_fixation(win, COLOR_BLINK)
    fixation_transition = make_fixation(win, COLOR_TRANSITION)

    trigger_patch = visual.Rect(
        win,
        width=TRIGGER_SIZE,
        height=TRIGGER_SIZE,
        pos=(-win_w / 2.0 + TRIGGER_SIZE / 2.0,
              win_h / 2.0 - TRIGGER_SIZE / 2.0),
        lineWidth=0,
        fillColor=rgb_255_to_psychopy(TRIGGER_RESET),
        fillColorSpace='rgb',
        lineColor=rgb_255_to_psychopy(TRIGGER_RESET),
        lineColorSpace='rgb',
    )

    # Immediately flip with the reset trigger to ensure the upper-left pixel
    # is black (value 0) from the first rendered frame, overriding the white
    # window background before any experiment content is shown.
    trigger_patch.draw()
    win.flip()

    # --- PsychoPy ExperimentHandler (instructor-recommended logging) ---
    exp_handler = ExperimentHandler(
        name='relational_meg',
        extraInfo=exp_info,
        dataFileName=str(data_dir / session_name),
    )

    # --- Manual CSV ---
    csv_path = data_dir / f'{session_name}.csv'
    manual_csv  = csv_path.open('w', newline='')
    manual_writer = csv.DictWriter(manual_csv, fieldnames=FIELDNAMES)
    manual_writer.writeheader()
    manual_csv.flush()

    # --- Start DIN log ---
    info = responsepixx_status(info, status='start')
    info = responsepixx_status(info, status='pause')   # pause until first trial

    try:
        show_text_screen(
            win,
            'Relational Reasoning Task\n\n'
            'You will see three shapes in sequence (the rule),\n'
            'then three more shapes (the test).\n\n'
            'Press  1  if the test follows the SAME pattern as the rule.\n'
            'Press  2  if it follows a DIFFERENT pattern.\n\n'
            'Keep your eyes on the fixation cross.\n'
            'Try not to blink during the shapes.\n'
            'When the cross turns BLUE, you may blink.\n\n'
            'Press any button to begin.',
            info,
            trigger_patch,
        )

        run_main_task(
            win, stimuli, stim_pos,
            fixation, fixation_correct, fixation_incorrect,
            fixation_blink, fixation_transition,
            trigger_patch, manual_writer, manual_csv,
            exp_handler, main_clock, info, sid, seed
        )

        show_text_screen(
            win,
            'The experiment is complete.\n\nThank you!\n\nPress any button to exit.',
            info,
            trigger_patch,
        )
        logging.info('Session finished normally.')

    except GracefulExit:
        logging.warning('Session aborted (Escape).')

    except Exception as e:
        logging.error(f'Session crashed: {repr(e)}')
        raise

    finally:
        try:
            manual_csv.close()
        finally:
            if not DUMMY_MODE:
                dp.DPxDisableDoutPixelMode()
                dp.DPxWriteRegCache()
                dp.DPxClose()
            win.close()
            core.quit()


if __name__ == '__main__':
    main()
