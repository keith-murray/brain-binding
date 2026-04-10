# Fixing the MEG script

## Background

We've ran the `relational/run_meg.py` on a participant in the MEG room, but the data was unusable because the reset trigger was wrong (that's my bad, I unknowingly changed it). Also, the white background would cause triggers to unintentionally go off because of the latent blue light signal in the white background. Also, the ISI and ITI minimum times are 0 ms, so it is hard to properly segment data.

## What to do

Let's change the background screen to black, and change the ISI and ITI minimum times to 500 ms. Also, let's make the blink window longer. The participant complained that it was too short.

Another issue that arose is that we could have more descriptive triggers for potential analyses. My thinking is this for triggers

- TRIGGER_RESET      = [0, 0,   0]
- TRIGGER_FIXATION   = [0, 0,   1]
- TRIGGER_RULE       = [0, 0,   2]   # all 3 rule stims share this code
- TRIGGER_TEST       = [0, 0,   4]   # First 2 test stims share this code
- TRIGGER_TEST_MATCH = [0, 0,   8]   # Last test stim when it did match (participant should report 1)
- TRIGGER_TEST_NON   = [0, 0,  16]   # Last test stim when it did not match (participant should report 2)
- TRIGGER_TRANSITION = [0, 0,  32]   # yellow fixation (rule→test boundary)
- TRIGGER_RESPONSE   = [0, 0,  64]   # when feedback case (but no differentiation)
- TRIGGER_BLINK      = [0, 0,  128]  # blue fixation (blink cue)
