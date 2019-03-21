Here we show an example of how the robot can visualize the preferences and constraints that it has learned.

The robot samples rewards from the latent space, and then compares the policies resulting from those sampled rewards. The robot visualizes states with low entropy (all the policies agree, teal) and high entropy (different rewards lead to different behavior, orange). Black cells show the constraint feature, and light gray / white cells show the two preferences.

Intuitively, the output should convey to the human "What can the robot learn?" In this example the robot can learn to move towards either light gray or white, but always avoids black cells.
