import uuid
from datetime import datetime
from pathlib import Path
import csv

import pandas as pd
from psychopy import core, visual, event, logging

PATH_TRIALS = 'trials.csv'


def main(sid: str) -> None:
    # -- CONFIG + LOGGING -- #
    main_clock = core.Clock()
    logging.setDefaultClock(main_clock)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    data_dir = Path(__file__).parent.parent / 'data' / timestamp
    data_dir.mkdir(parents=True, exist_ok=True)

    log_path = data_dir / f'{sid}.log'
    logging.LogFile(str(log_path), level=logging.INFO, filemode='w')
    logging.info(f'Starting session sid={sid}')

    # -- WINDOW -- #
    win = visual.Window(size=[1200, 1000], fullscr=False, color='grey', name='Window', screen=1)

    # -- STIMULUS TIMELINE -- #
    fixation = visual.TextStim(win=win, pos=(0, 0), text='+', color='white')
    fixation_duration = 0.7

    stroop = visual.TextStim(win=win, pos=(0, 0), text='')
    stroop_duration = 2.5  # response deadline (seconds)

    iti = visual.TextStim(win=win, pos=(0, 0), text='')
    iti_duration = 0.5

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
    out_csv_path = data_dir / f'{sid}.csv'
    f = out_csv_path.open('w', newline='')
    fieldnames = ['sid', 'trial', 'color', 'word', 'key', 'rt', 'correct']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.flush()

    rt_clock = core.Clock()

    try:
        for t_idx, trial in enumerate(trials.itertuples(index=False)):
            color = str(trial.color)
            word = str(trial.word)

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
            rt_clock.reset()

            # Response (robust polling loop)
            result = event.waitKeys(
                keyList=list(key_mapping.keys()),
                timeStamped=rt_clock,
                maxWait=stroop_duration,
                clearEvents=True
            )

            if result is not None:
                key, rt = result[0]
                correct = (key_mapping.get(key) == color)
                logging.data(f'Trial {t_idx} response: key={key}, rt={rt:.4f}, correct={correct}')
            else:
                key, rt, correct = None, None, False
                logging.warning(f'Trial {t_idx} timeout (no response within {stroop_duration}s)')

            writer.writerow(
                {
                    'sid': sid,
                    'trial': t_idx,
                    'color': color,
                    'word': word,
                    'key': key,
                    'rt': rt,
                    'correct': correct,
                }
            )
            f.flush()

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