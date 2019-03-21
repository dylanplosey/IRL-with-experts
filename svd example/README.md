Example that shows that using a prior based SVD leads to faster learning than standard.

I use the same example as in the front figure, and compare a uniform prior, a "Euclidean" (here referred to as baseline) prior, and our SVD approach. The figures show that the SVD approach is robust to an unintended demonstration (does not learn to go to the black area, which is a constraint), but also learns the human's preference from an intended demonstration (the white cell is the current user's preference, and the gray cell is another preference which is more common in the expert data).

I also print out the overall regret and the state where each prior results in the worst action. Summary: uniform doesn't recognize that the constraint is important, while baseline sticks with the expert demonstrations, where the gray cell is a more common preference. SVD is robust to unintended demonstrations but learns quickly from a preference-related demonstration!
