# Incorporating MEG experimental design feedback

## Background

We have implemented the relational reasoning task for the MEG setup and the course instructors have done an initial test. Here is there feedback.

## Instructor feedback

1) Your blink window doesn't send a trigger. Show_blink_window() fires TRIGGER_RESET. So the blink window is invisible in the MEG data. You should mark that if you ever want to exclude blink windows.

2) The triggers use powers of 2, so they should be:
[0, 0, 1]
[0, 0, 2]
[0, 0, 4]
[0, 0, 8]
[0, 0, 16]
...
Up to 128.

### Major comment

I think you should record the subject button presses to the psychopy log so that you can easily parse by accuracy, RT, etc.  Unfortunately this isn't as simple as logging a keyboard press.  I'm attaching here the code you need to implement into your script.  Once you do, send it back our way and we can test it.  It's pretty well commented as you'll see.

## My interpretation

Changing the triggers is the most important part, and it should be quite easy. I think that we should have a trigger every time the color of the fixation cross changes. This means that instead of having a trigger for each of the different rule and test stimuli, just have one trigger for rule stimuli and one trigger for test stimuli. We can simply deduce based on the sequence of triggers what the position of each stimulus is.

As for the major comment, we already create a CSV log for human performance per trial but the instructor did not see this part. Let's just also implement the logging that the instructor recommends in tandem with our already established logging.
