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

Trigger Scheme (VPIXX Pixel Mode Blue, values 0–7):
    - Reset    (0): RGB = (0, 0, 0)
    - Fixation (1): RGB = (0, 0, 1)
    - Rule 1   (2): RGB = (0, 0, 2)
    - Rule 2   (3): RGB = (0, 0, 3)
    - Rule 3   (4): RGB = (0, 0, 4)
    - Test 1   (5): RGB = (0, 0, 5)
    - Test 2   (6): RGB = (0, 0, 6)
    - Test 3   (7): RGB = (0, 0, 7)

Blink Trials:
    - Blue fixation cross after each trial's response/feedback
    - Subject blinks freely during this period
    - Keeps blink artifacts outside the trial window

Output:
    - CSV log per session (BIDS-style naming)
    - PsychoPy log file
"""

import csv
import random
from datetime import datetime
from itertools import permutations
from pathlib import Path

import numpy as np
from psychopy import core, data, event, gui, logging, visual

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
FIXATION_DURATION  = 0.500
STIM_DURATION      = 0.500
FEEDBACK_DURATION  = 0.500
BLINK_DURATION     = 1.500
RESPONSE_DEADLINE  = 2.000

ISI_MEAN            = 1.0
ISI_SD              = 0.2
ISI_MIN             = 0.0
TRANSITION_DURATION = 1.5   # fixed gap between rule and test phases

ITI_BASE = 3.0
ITI_SD   = 1.0
ITI_MIN  = 0.0

# Task structure
N_RUNS         = 4
TRIALS_PER_RUN = 30   # 4 × 30 = 120 total
RULE_REPS      = 5    # 24 configs × 5 = 120

RESPONSE_KEYS = ['1', '2']  # 1 = match, 2 = no-match

# Triggers (VPIXX Pixel Mode Blue channel, values 0–7)
TRIGGER_SIZE     = 1           # pixels
TRIGGER_RESET    = [0, 0, 0]
TRIGGER_FIXATION = [0, 0, 1]
TRIGGER_RULE1    = [0, 0, 2]
TRIGGER_RULE2    = [0, 0, 3]
TRIGGER_RULE3    = [0, 0, 4]
TRIGGER_TEST1    = [0, 0, 5]
TRIGGER_TEST2    = [0, 0, 6]
TRIGGER_TEST3    = [0, 0, 7]

# Fixation cross colors
COLOR_FIXATION    = 'black'
COLOR_CORRECT     = 'green'
COLOR_INCORRECT   = 'red'
COLOR_BLINK       = 'blue'
COLOR_TRANSITION  = 'yellow'

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
        vertices=((0, -30), (0, 30), (0, 0), (-30, 0), (30, 0)),
        lineWidth=10,
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
    base = configs * RULE_REPS   # 120 items
    rng.shuffle(base)

    # Balanced match labels
    match_labels = [True] * 60 + [False] * 60
    rng.shuffle(match_labels)

    # Balanced mismatch-type labels (consumed only for mismatch trials)
    mismatch_types = ['rule_order'] * 30 + ['random'] * 30
    rng.shuffle(mismatch_types)
    mismatch_iter = iter(mismatch_types)

    trials = []
    for cfg, is_match in zip(base, match_labels):
        a, b, rule_type = cfg['A_stim'], cfg['B_stim'], cfg['rule_type']
        rule_seq = [a, b, a] if rule_type == 'ABA' else [a, b, b]

        # Test pair: X ≠ Y, chosen independently of A, B
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
            else:  # random
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
    """Draw image stimulus with trigger; return onset time.
    Does NOT flip to blank after — caller handles the next state.
    """
    stim.pos = stim_pos
    stim.draw()
    _send_trigger(trigger_patch, trigger_color)
    trigger_patch.draw()
    t = win.flip()
    core.wait(STIM_DURATION)
    return t


def show_blank(win: visual.Window, fixation: visual.ShapeStim,
               trigger_patch: visual.Rect, duration: float) -> None:
    """Show fixation with reset trigger for given duration."""
    fixation.draw()
    _send_trigger(trigger_patch, TRIGGER_RESET)
    trigger_patch.draw()
    win.flip()
    if duration > 0:
        interruptible_wait(duration)


def show_response_window(win: visual.Window, fixation: visual.ShapeStim,
                         trigger_patch: visual.Rect,
                         rt_clock: core.Clock) -> tuple:
    """Show fixation; wait for response. Return (t_onset, waitKeys result)."""
    fixation.draw()
    _send_trigger(trigger_patch, TRIGGER_RESET)
    trigger_patch.draw()
    t = win.flip()
    rt_clock.reset()
    result = event.waitKeys(
        keyList=RESPONSE_KEYS + ['escape'],
        timeStamped=rt_clock,
        maxWait=RESPONSE_DEADLINE,
        clearEvents=True,
    )
    if result and result[0][0] == 'escape':
        raise GracefulExit()
    return t, result


def show_feedback(win: visual.Window,
                  fixation_correct: visual.ShapeStim,
                  fixation_incorrect: visual.ShapeStim,
                  trigger_patch: visual.Rect,
                  correct: bool) -> None:
    """Briefly color the fixation cross blue (correct) or red (incorrect)."""
    fix = fixation_correct if correct else fixation_incorrect
    fix.draw()
    _send_trigger(trigger_patch, TRIGGER_RESET)
    trigger_patch.draw()
    win.flip()
    core.wait(FEEDBACK_DURATION)


def show_blink_window(win: visual.Window, fixation_blink: visual.ShapeStim,
                      trigger_patch: visual.Rect) -> None:
    """Show blue fixation cross to cue blinking."""
    fixation_blink.draw()
    _send_trigger(trigger_patch, TRIGGER_RESET)
    trigger_patch.draw()
    win.flip()
    core.wait(BLINK_DURATION)


def show_text_screen(win: visual.Window, text: str) -> None:
    """Show a text message and wait for 1 or 2."""
    msg = visual.TextStim(
        win=win,
        text=text,
        color='black',
        height=30,
        wrapWidth=win.size[0] * 0.8,
    )
    msg.draw()
    win.flip()
    keys = event.waitKeys(keyList=['1', '2', 'escape'])
    if 'escape' in keys:
        raise GracefulExit()


# =============================================================================
# TASK RUNNER
# =============================================================================

def run_main_task(win, stimuli, stim_pos,
                  fixation, fixation_correct, fixation_incorrect,
                  fixation_blink, fixation_transition,
                  trigger_patch, writer, csv_file,
                  main_clock, rt_clock, sid, seed):

    trials = gen_main_trials(seed)

    for run_idx in range(N_RUNS):
        if run_idx > 0:
            show_text_screen(
                win,
                f'Break — run {run_idx} of {N_RUNS} complete.\n\n'
                'Rest for a moment, then press 1 or 2 to continue.'
            )

        run_trials = trials[run_idx * TRIALS_PER_RUN:(run_idx + 1) * TRIALS_PER_RUN]

        for t_rel, trial in enumerate(run_trials):
            t_idx = run_idx * TRIALS_PER_RUN + t_rel

            # Check for escape at trial boundary
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
                                    TRIGGER_RULE1, trigger_patch)
            show_blank(win, fixation, trigger_patch, isi1)

            t_rule2 = show_stimulus(win, stimuli[rule_seq[1]], stim_pos,
                                    TRIGGER_RULE2, trigger_patch)
            show_blank(win, fixation, trigger_patch, isi2)

            t_rule3 = show_stimulus(win, stimuli[rule_seq[2]], stim_pos,
                                    TRIGGER_RULE3, trigger_patch)
            show_blank(win, fixation_transition, trigger_patch, isi3)

            # 3. Test phase
            t_test1 = show_stimulus(win, stimuli[test_seq[0]], stim_pos,
                                    TRIGGER_TEST1, trigger_patch)
            show_blank(win, fixation, trigger_patch, isi4)

            t_test2 = show_stimulus(win, stimuli[test_seq[1]], stim_pos,
                                    TRIGGER_TEST2, trigger_patch)
            show_blank(win, fixation, trigger_patch, isi5)

            t_test3 = show_stimulus(win, stimuli[test_seq[2]], stim_pos,
                                    TRIGGER_TEST3, trigger_patch)

            # 4. Response window (fixation with reset trigger)
            t_response, result = show_response_window(
                win, fixation, trigger_patch, rt_clock
            )

            if result is not None:
                response_key, rt = result[0]
                correct = int((response_key == '1') == bool(trial['match']))
            else:
                response_key = rt = None
                correct = 0

            # 5. Feedback
            show_feedback(win, fixation_correct, fixation_incorrect,
                          trigger_patch, bool(correct))

            # 6. Blink window
            show_blink_window(win, fixation_blink, trigger_patch)

            # 7. ITI
            show_blank(win, fixation, trigger_patch, iti_dur)

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
            writer.writerow(row)
            csv_file.flush()


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
    sid = exp_info['participant']

    session_name = (f"sub-{sid}"
                    f"_ses-{exp_info['session']}"
                    f"_task-relational")
    data_dir = Path(__file__).parent.parent / 'data' / session_name
    data_dir.mkdir(parents=True, exist_ok=True)

    main_clock = core.Clock()
    logging.setDefaultClock(main_clock)
    log_path = data_dir / f'{session_name}.log'
    logging.LogFile(str(log_path), level=logging.INFO, filemode='w')
    logging.info(f'sid={sid}, seed={seed}')

    win = visual.Window(
        fullscr=True,
        color='white',
        colorSpace='rgb',
        units='pix',
        allowGUI=False,
        screen=0,   # change if projector is on a different display
    )
    win_w, win_h = win.size

    # Stimulus size and position scaled to screen
    # Equivalent to (0.28, 0.37) norm and (0, 0.15) norm
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

    # Trigger patch: 1×1 pixel in upper-left corner
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

    rt_clock = core.Clock()

    csv_path = data_dir / f'{session_name}.csv'
    csv_file = csv_path.open('w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    writer.writeheader()
    csv_file.flush()

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
            'Press 1 or 2 to begin.'
        )

        run_main_task(
            win, stimuli, stim_pos,
            fixation, fixation_correct, fixation_incorrect,
            fixation_blink, fixation_transition,
            trigger_patch, writer, csv_file,
            main_clock, rt_clock, sid, seed
        )

        show_text_screen(
            win,
            'The experiment is complete.\n\nThank you!\n\nPress 1 or 2 to exit.'
        )
        logging.info('Session finished normally.')

    except GracefulExit:
        logging.warning('Session aborted (Escape).')

    except Exception as e:
        logging.error(f'Session crashed: {repr(e)}')
        raise

    finally:
        try:
            csv_file.close()
        finally:
            win.close()
            core.quit()


if __name__ == '__main__':
    main()
