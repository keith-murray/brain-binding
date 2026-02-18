import uuid
from pathlib import Path
import csv

import pandas as pd
from psychopy import core, visual, event, logging

PATH_TRIALS = 'trials.csv'


def main(sid: str) -> None:
    # -- CONFIG + LOGGING -- #
    main_clock = core.Clock()
    logging.setDefaultClock(main_clock)

    log_path = Path(f'{sid}.log')
    logging.LogFile(str(log_path), level=logging.INFO, filemode='w')
    logging.info(f'Starting session sid={sid}')

    # -- WINDOW -- #
    win = visual.Window(size=[1200, 1000], fullscr=False, color='black', name='Window')

    # -- STIMULUS TIMELINE -- #
    fixation = visual.TextStim(win=win, pos=(0, 0), text='+', color='white')
    fixation_duration = 1.

    stroop = visual.TextStim(win=win, pos=(0, 0), text='')
    stroop_duration = 2.4  # response deadline (seconds)

    iti = visual.TextStim(win=win, pos=(0, 0), text='')
    iti_duration = 1.

    # -- KEY MAPPING -- #
    key_mapping = {
        'd': 'red',
        'f': 'blue',
        'j': 'green',
        'k': 'yellow',
    }

    # -- TRIALS -- #
    trials = pd.read_csv(PATH_TRIALS)

    # -- CRASH-SAFE CSV WRITER -- #
    out_csv_path = Path(f'{sid}.csv')
    f = out_csv_path.open('w', newline='')

    f.flush()

    rt_clock = core.Clock()
    instruction = visual.TextStim(text='Say the COLOR as fast as you can!', win=win, color='white')
    instruction.draw()
    win.flip()

    event.waitKeys(keyList=['space'], clearEvents=True)

    try:
        for t_idx, trial in enumerate(trials.itertuples(index=False)):
            color = str(trial.color)
            word = str(trial.word)
            stroop_duration = stroop_duration - .2
            stroop_duration = max(stroop_duration, .4)
            fixation_duration = fixation_duration - .1
            fixation_duration = max(fixation_duration, .3)
            iti_duration = iti_duration - .1
            iti_duration = max(iti_duration, .3)
            if color != 'red':
                stroop_duration = 2.
                fixation_duration = 1.
                iti_duration = 1.

            logging.data(f'Trial {t_idx} start: color={color}, word={word}')

            # Fixation
            fixation.draw()
            win.flip()
            core.wait(fixation_duration)

            # Stroop onset
            stroop.text = word
            stroop.color = color
            stroop.draw()
            win.flip()

            # Response (robust polling loop)
            event.waitKeys(
                keyList=list(key_mapping.keys()),
                timeStamped=rt_clock,
                maxWait=stroop_duration,
                clearEvents=True
            )

            # ITI
            iti.draw()
            win.flip()
            core.wait(iti_duration)

        logging.info('Session finished normally.')

    except Exception as e:
        logging.error(f'Session crashed: {repr(e)}')
        raise

    finally:
        try:
            f.close()
        finally:
            win.close()
            core.quit()


if __name__ == '__main__':
    subject_id = str(uuid.uuid4())
    main(sid=subject_id)
