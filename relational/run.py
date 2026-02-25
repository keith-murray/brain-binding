import csv
import random
import uuid
from datetime import datetime
from itertools import permutations
from pathlib import Path

import numpy as np
from psychopy import core, event, logging, visual

# --- Constants ---

STIM_NAMES = ['circle', 'rectangle', 'star', 'triangle']
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

WIN_SIZE = [1200, 1000]
WIN_UNITS = 'norm'
WIN_SCREEN = 0

STIM_SIZE = (0.28, 0.37)
STIM_POS = (0, 0.15)

SLOT_X = [-0.65, -0.22, 0.22, 0.65]
SLOT_Y = 0.0
LABEL_Y = -0.28

RESPONSE_KEYS = ['1', '2', '3', '4']
RESPONSE_DEADLINE = 2.0

FIXATION_DURATION = 0.5
STIM_DURATION = 0.5

# WM task
WM_ISI_CONDITIONS = [1.0, 2.5]
WM_REPS = 3  # per stimulus per ISI → 4 × 2 × 3 = 24

# Main task
N_RUNS = 4
TRIALS_PER_RUN = 30  # 4 × 30 = 120
RULE_REPS = 5  # 24 configs × 5 = 120

ISI_MEAN = 1.0
ISI_SD = 0.2
ISI_MIN = 0.0

ITI_BASE = 3.0
ITI_SD = 1.0
ITI_MIN = 0.0

REST_DURATION = 120.0  # 2 minutes

MAX_STREAK = 3

FIELDNAMES = [
    'block', 'sid', 'trial',
    'sample_stim', 'isi_condition',
    'rule_type', 'A_stim', 'B_stim', 'A_prime', 'B_prime', 'rule_sequence', 'test_sequence',
    'correct_next_stim', 'slot_mapping', 'correct_stim',
    'response_key', 'response_slot', 'response_stim', 'correct', 'rt',
    'isi1', 'isi2', 'isi3', 'isi4', 'iti',
    't_fixation', 't_sample', 't_delay',
    't_rule1', 't_rule2', 't_rule3', 't_test1', 't_test2', 't_response',
]


# --- Setup helpers ---

def make_window() -> visual.Window:
    return visual.Window(
        size=WIN_SIZE,
        fullscr=False,
        color='grey',
        units=WIN_UNITS,
        name='Window',
        screen=WIN_SCREEN,
    )


def load_stimuli(win: visual.Window) -> dict:
    stimuli = {}
    for name in STIM_NAMES:
        stimuli[name] = visual.ImageStim(
            win=win,
            image=str(ASSETS_DIR / f'{name}.png'),
            size=STIM_SIZE,
            units=WIN_UNITS,
        )
    return stimuli


def make_slot_mapping() -> list:
    """Return a random permutation of STIM_NAMES for the 4 response slots."""
    mapping = STIM_NAMES.copy()
    random.shuffle(mapping)
    return mapping


def jitter(mean: float, sd: float, min_val: float) -> float:
    rng = np.random.default_rng()
    return float(max(min_val, mean + rng.normal(0, sd)))


# --- Trial generation ---


def gen_wm_trials(seed: int) -> list:
    """24 trials: 4 stims × 2 ISI conditions × 3 reps, seeded shuffle."""
    trials = [
        {'sample_stim': stim, 'isi_condition': isi}
        for stim in STIM_NAMES
        for isi in WM_ISI_CONDITIONS
        for _ in range(WM_REPS)
    ]
    random.Random(seed).shuffle(trials)
    return trials


def gen_main_trials(seed: int) -> list:
    """120 trials: 24 (A,B,rule) configs × 5 reps, max-streak-3 on rule_type.

    Uses a greedy construction: shuffle indices, then place items one by one
    picking the first valid item from the remaining pool.  With a balanced
    60/60 ABA/ABB split this always succeeds in a small number of restarts.
    """
    configs = [
        {'A_stim': a, 'B_stim': b, 'rule_type': rule}
        for a, b in permutations(STIM_NAMES, 2)
        for rule in ['ABA', 'ABB']
    ]
    base = configs * RULE_REPS  # 120 items

    rng = random.Random(seed)
    indices = list(range(len(base)))
    result = None

    for _ in range(200):
        rng.shuffle(indices)
        placed = []
        pool = list(indices)  # remaining indices in shuffled order

        while pool:
            # If the last MAX_STREAK placements are all the same rule, forbid it
            forbidden = None
            if len(placed) >= MAX_STREAK:
                tail = {base[placed[-(i + 1)]]['rule_type'] for i in range(MAX_STREAK)}
                if len(tail) == 1:
                    forbidden = next(iter(tail))

            # Pick the first item from the (shuffled) pool that is not forbidden
            found = -1
            for pos, idx in enumerate(pool):
                if forbidden is None or base[idx]['rule_type'] != forbidden:
                    found = pos
                    break

            if found == -1:
                break  # stuck — restart with a new shuffle
            placed.append(pool.pop(found))

        if len(placed) == len(base):
            result = placed
            break

    if result is None:
        result = indices  # fallback (should not occur with balanced data)

    trials = []
    for idx in result:
        cfg = base[idx]
        a, b, rule = cfg['A_stim'], cfg['B_stim'], cfg['rule_type']
        if rule == 'ABA':
            rule_seq = [a, b, a]
        else:
            rule_seq = [a, b, b]
        a_prime, b_prime = rng.sample(STIM_NAMES, 2)  # A' ≠ B', can overlap with A or B
        correct = a_prime if rule == 'ABA' else b_prime
        trials.append({
            'rule_type': rule,
            'A_stim': a,
            'B_stim': b,
            'A_prime': a_prime,
            'B_prime': b_prime,
            'rule_sequence': ','.join(rule_seq),
            'test_sequence': f'{a_prime},{b_prime}',
            'correct_next_stim': correct,
        })
    return trials


# --- Display helpers ---

def show_fixation(win: visual.Window, fixation: visual.TextStim,
                  duration: float, clock: core.Clock) -> float:
    fixation.draw()
    win.flip()
    t_onset = clock.getTime()
    core.wait(duration)
    return t_onset


def show_stimulus(win: visual.Window, stim: visual.ImageStim,
                  duration: float, clock: core.Clock) -> float:
    stim.pos = STIM_POS
    stim.draw()
    win.flip()
    t_onset = clock.getTime()
    core.wait(duration)
    return t_onset


def show_blank(win: visual.Window, fixation: visual.TextStim,
               duration: float, clock: core.Clock) -> float:
    """Show fixation cross for ISI/ITI."""
    fixation.draw()
    win.flip()
    t_onset = clock.getTime()
    core.wait(duration)
    return t_onset


def show_response(win: visual.Window, stimuli: dict, slot_mapping: list,
                  key_labels: list, clock: core.Clock,
                  rt_clock: core.Clock) -> tuple:
    """Draw 4AFC response screen; return (t_onset, waitKeys result)."""
    for i, stim_name in enumerate(slot_mapping):
        stimuli[stim_name].pos = (SLOT_X[i], SLOT_Y)
        stimuli[stim_name].draw()
    for label in key_labels:
        label.draw()
    win.flip()
    t_onset = clock.getTime()
    rt_clock.reset()
    result = event.waitKeys(
        keyList=RESPONSE_KEYS,
        timeStamped=rt_clock,
        maxWait=RESPONSE_DEADLINE,
        clearEvents=True,
    )
    return t_onset, result


def show_text_screen(win: visual.Window, text: str) -> None:
    """Show a text message and wait for spacebar."""
    msg = visual.TextStim(
        win=win,
        text=text,
        color='white',
        height=0.07,
        wrapWidth=1.8,
    )
    msg.draw()
    win.flip()
    event.waitKeys(keyList=['space'])


# --- Task runners ---

def run_wm_task(win, stimuli, fixation, key_labels, writer, csv_file,
                main_clock, rt_clock, sid, seed):
    trials = gen_wm_trials(seed)

    for t_idx, trial in enumerate(trials):
        sample_stim = trial['sample_stim']
        isi_cond = trial['isi_condition']

        logging.data(f'WM trial {t_idx}: sample={sample_stim}, isi={isi_cond}')

        t_fixation = show_fixation(win, fixation, FIXATION_DURATION, main_clock)
        t_sample = show_stimulus(win, stimuli[sample_stim], STIM_DURATION, main_clock)
        t_delay = show_blank(win, fixation, isi_cond, main_clock)

        slot_mapping = make_slot_mapping()
        t_response, result = show_response(
            win, stimuli, slot_mapping, key_labels, main_clock, rt_clock
        )

        if result is not None:
            response_key, rt = result[0]
            response_slot = int(response_key)
            response_stim = slot_mapping[response_slot - 1]
            correct = int(response_stim == sample_stim)
            logging.data(
                f'WM trial {t_idx} response: key={response_key}, '
                f'rt={rt:.4f}, correct={correct}'
            )
        else:
            response_key = rt = response_slot = response_stim = None
            correct = 0
            logging.warning(f'WM trial {t_idx} timeout')

        row = {fn: '' for fn in FIELDNAMES}
        row.update({
            'block': 'WM',
            'sid': sid,
            'trial': t_idx,
            'sample_stim': sample_stim,
            'isi_condition': isi_cond,
            'slot_mapping': ','.join(slot_mapping),
            'correct_stim': sample_stim,
            'response_key': response_key,
            'response_slot': response_slot,
            'response_stim': response_stim,
            'correct': correct,
            'rt': rt,
            't_fixation': t_fixation,
            't_sample': t_sample,
            't_delay': t_delay,
            't_response': t_response,
        })
        writer.writerow(row)
        csv_file.flush()


def run_rest(win, fixation):
    fixation.draw()
    win.flip()
    core.wait(REST_DURATION)


def run_main_task(win, stimuli, fixation, key_labels, writer, csv_file,
                  main_clock, rt_clock, sid, seed):
    trials = gen_main_trials(seed)

    for run_idx in range(N_RUNS):
        if run_idx > 0:
            show_text_screen(
                win,
                f'Break — run {run_idx} of {N_RUNS} complete.\n\n'
                'Rest for a moment, then press SPACE to continue.'
            )

        run_trials = trials[run_idx * TRIALS_PER_RUN:(run_idx + 1) * TRIALS_PER_RUN]

        for t_rel, trial in enumerate(run_trials):
            t_idx = run_idx * TRIALS_PER_RUN + t_rel

            rule_type = trial['rule_type']
            a_stim = trial['A_stim']
            b_stim = trial['B_stim']
            a_prime = trial['A_prime']
            b_prime = trial['B_prime']
            rule_seq = trial['rule_sequence'].split(',')
            test_seq = trial['test_sequence'].split(',')
            correct_next = trial['correct_next_stim']

            logging.data(
                f'Main trial {t_idx} run {run_idx + 1}: '
                f'rule={rule_type}, A={a_stim}, B={b_stim}, A\'={a_prime}, B\'={b_prime}'
            )

            isi1 = jitter(ISI_MEAN, ISI_SD, ISI_MIN)
            isi2 = jitter(ISI_MEAN, ISI_SD, ISI_MIN)
            isi3 = jitter(ISI_MEAN, ISI_SD, ISI_MIN)
            isi4 = jitter(ISI_MEAN, ISI_SD, ISI_MIN)
            iti_dur = jitter(ITI_BASE, ITI_SD, ITI_MIN)

            # 1. Trial-start fixation
            t_fixation = show_fixation(win, fixation, FIXATION_DURATION, main_clock)

            # 2. Rule phase
            t_rule1 = show_stimulus(win, stimuli[rule_seq[0]], STIM_DURATION, main_clock)
            show_blank(win, fixation, isi1, main_clock)
            t_rule2 = show_stimulus(win, stimuli[rule_seq[1]], STIM_DURATION, main_clock)
            show_blank(win, fixation, isi2, main_clock)
            t_rule3 = show_stimulus(win, stimuli[rule_seq[2]], STIM_DURATION, main_clock)
            show_blank(win, fixation, isi3, main_clock)

            # 3. Test phase
            t_test1 = show_stimulus(win, stimuli[test_seq[0]], STIM_DURATION, main_clock)
            show_blank(win, fixation, isi4, main_clock)
            t_test2 = show_stimulus(win, stimuli[test_seq[1]], STIM_DURATION, main_clock)

            # 4. Response screen
            slot_mapping = make_slot_mapping()
            t_response, result = show_response(
                win, stimuli, slot_mapping, key_labels, main_clock, rt_clock
            )

            if result is not None:
                response_key, rt = result[0]
                response_slot = int(response_key)
                response_stim = slot_mapping[response_slot - 1]
                correct = int(response_stim == correct_next)
                logging.data(
                    f'Main trial {t_idx} response: key={response_key}, '
                    f'rt={rt:.4f}, correct={correct}'
                )
            else:
                response_key = rt = response_slot = response_stim = None
                correct = 0
                logging.warning(f'Main trial {t_idx} timeout')

            # 5. ITI fixation
            show_blank(win, fixation, iti_dur, main_clock)

            row = {fn: '' for fn in FIELDNAMES}
            row.update({
                'block': f'main_run{run_idx + 1}',
                'sid': sid,
                'trial': t_idx,
                'rule_type': rule_type,
                'A_stim': a_stim,
                'B_stim': b_stim,
                'A_prime': a_prime,
                'B_prime': b_prime,
                'rule_sequence': trial['rule_sequence'],
                'test_sequence': trial['test_sequence'],
                'correct_next_stim': correct_next,
                'slot_mapping': ','.join(slot_mapping),
                'correct_stim': correct_next,
                'response_key': response_key,
                'response_slot': response_slot,
                'response_stim': response_stim,
                'correct': correct,
                'rt': rt,
                'isi1': isi1,
                'isi2': isi2,
                'isi3': isi3,
                'isi4': isi4,
                'iti': iti_dur,
                't_fixation': t_fixation,
                't_rule1': t_rule1,
                't_rule2': t_rule2,
                't_rule3': t_rule3,
                't_test1': t_test1,
                't_test2': t_test2,
                't_response': t_response,
            })
            writer.writerow(row)
            csv_file.flush()


# --- Entry point ---

def main(sid: str) -> None:
    seed = int(datetime.now().timestamp() * 1000) % (2 ** 31)

    main_clock = core.Clock()
    logging.setDefaultClock(main_clock)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    data_dir = Path(__file__).parent.parent / 'data' / timestamp
    data_dir.mkdir(parents=True, exist_ok=True)

    log_path = data_dir / f'{sid}.log'
    logging.LogFile(str(log_path), level=logging.INFO, filemode='w')
    logging.info(f'Starting session sid={sid}, seed={seed}')

    win = make_window()
    stimuli = load_stimuli(win)
    fixation = visual.TextStim(win=win, pos=(0, 0), text='+', color='white', height=0.07)

    # Pre-create key labels (static across trials)
    key_labels = [
        visual.TextStim(win=win, text=key, pos=(SLOT_X[i], LABEL_Y),
                        color='white', height=0.07)
        for i, key in enumerate(RESPONSE_KEYS)
    ]

    rt_clock = core.Clock()

    out_csv_path = data_dir / f'{sid}.csv'
    csv_file = out_csv_path.open('w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    writer.writeheader()
    csv_file.flush()

    try:
        # Block 1: WM task
        show_text_screen(
            win,
            'Working Memory Task\n\n'
            'You will see a shape, then pick the matching shape from 4 options.\n\n'
            'Press SPACE to begin.'
        )
        run_wm_task(win, stimuli, fixation, key_labels, writer, csv_file,
                    main_clock, rt_clock, sid, seed)

        # Block 2: Rest
        show_text_screen(
            win,
            'Rest (2 minutes)\n\n'
            'Please fixate on the cross and relax.\n\n'
            'Press SPACE to begin rest.'
        )
        run_rest(win, fixation)

        # Block 3: Main task
        show_text_screen(
            win,
            'Main Task\n\n'
            'You will see a sequence of shapes that follow a pattern.\n'
            'Then you will see the start of a new sequence.\n'
            'Choose the shape that comes next.\n\n'
            'Press SPACE to begin.'
        )
        run_main_task(win, stimuli, fixation, key_labels, writer, csv_file,
                      main_clock, rt_clock, sid, seed + 1)

        # End screen
        show_text_screen(win, 'The experiment is complete. Thank you!\n\nPress SPACE to exit.')

        logging.info('Session finished normally.')

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
    subject_id = str(uuid.uuid4())
    main(sid=subject_id)
