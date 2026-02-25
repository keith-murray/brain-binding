# PsychoPy Interaction with Scanner

## Background

The task works well, but we need to get the task to interface with the fMRI scanner. Particularly, we want the task to be in sync with the TR of the fMRI scanner. The scanner will send the computer a notification of a TR via the = sign button. We want to record these

## Changes to the task

### Working memory block

At the beginning of the first block, when the screen appears that says "You will see a shape, then pick the matching shape from 4 options.", the participant should then be able to press any button to continue. Then our program should wait for the = sign button to begin.

### Rest block

Let's remove this.

### Main task

At the beginning of this block, when the screen appears that says "You will see a sequence of shapes that follow a pattern", the participant should then be able to press any button to continue. Then our program should wait for the = sign button to begin.
