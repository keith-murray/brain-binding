# Visualizing MEG data for the third test stimulus

## Background

Looking at the behavioral data for our MEG experiment, we saw that reaction times for "random" trials were much faster than "match" and "rule order" trials. This suggests that something interesting might be happening when the participant sees the third stimulus in the test sequence. This stimulus is key as it clues the participant in as to whether the trial is match or non-match.

## Visualizing MEG data

Like all good analyses, once we know what part of the experiment is interesting, we should just visualize that data. Specifically, I want to visualize magnetometers for the time course around the presentation of the third test stimulus. The MNE visualizer is especially good for this, but our triggers of `test_match` and `test_non_match` do not separate between `test_non_match_random` and `test_non_match_rule_order`, so we need to figure out a way to partition that data. We might have to cross reference the behavioral file at `data\2026-04-10\sub-001_events.csv` to know what trials correspond to what type of non-match.
