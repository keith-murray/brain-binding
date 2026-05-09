what's currently happening: 
  -
  - neural rdm: for each epoch, we are doing some distance function between condition-mean patterns. that is we are averaging all the trials per condition (on each timestep for time-resolved) adn then taking the distance between these centroids. 
  - we are doing so in an epoch-resolved way, in which we take all the rule-phase epochs (or others) as the same. 


What I would like to see: 
  - same neural rdm computation is fine.
  - but for the conditioning, you need to take a trial-centric view as opposed to an epoch-centric view, as each epoch per trial (substage) is inherently different and can't be averaged. Here is what I want you to do: 
