# Behavior analysis figure

## Background

My group is writing a report of our fMRI results. We want to have a summary figure for the behavior data. Look at the `behavior/fmri/behavioral_analysis/behavior_analysis.py` script for context about what we've done.

## The figure to make

I want a 2x3 panel figure. Here is what should be included in each panel:

- Panel (0,0) — Bar plot of overall accuracy separated based on participant. Give each participant a color, label the color in this panel, and display the legend. This is the only location where we will display the legend mapping each participant to their color.
- Panel (0,1) — Bar plot of accuracy by rule type, ABA and ABB. There will be two groups of bar plots, one for ABA and one for ABB. In one group, there will be separate bars for each participant, but the bars should be touching. Again, color participant data based on the color scheme from panel (0,0).
- Panel (0,2) — Bar plot of accuracy by repeat vs switch trials. Refer to `fig1_rt_repeat_vs_switch.png` for context. Again, two groups of bars for each condition, repeat or switch. In one group, there will be separate bars for each participant, but the bars should be touching. Again, color participant data based on the color scheme from panel (0,0).
- Panel (1,0) — Reaction time distribution for correct trials. There should be three different reaction time distributions for each participant. Label the median for each participant. If possible, fit a drift-diffusion model (DDM) and plot the fitted curve.
- Panel (1,1) — Reaction time distribution for rule types ABA and ABB. Aggregate data from across participants. Label the median for each participant. If possible, fit a drift-diffusion model (DDM) and plot the fitted curve.
- Panel (1,2) — Reaction time distribution for repeat vs switch trials. Aggregate data from across participants. Label the median for each participant. If possible, fit a drift-diffusion model (DDM) and plot the fitted curve.

## Question

Is there an opportunity for statistics anywhere in this plot? We don't have to plot them, but could include p-values in the caption of the figure?
