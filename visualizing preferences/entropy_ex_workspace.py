import math
import operator
import entropy_ex_utils as ut
import numpy as np
import matplotlib.pyplot as plt


# collect expert weights offline
W = ut.generate_experts()

# find latent representation
theta_bar = np.mean(W, axis=1)
W_zero = W - np.asarray([theta_bar] * 10).T
U, S, Vt = np.linalg.svd(W_zero)
Sm = np.diag(S)
L = np.dot(U, Sm[:, [0, 1]])

# pick a gridworld
nFeats = 3
nRows = 8
nCols = 8
gamma = 0.5
mdp = ut.GridWorld(nFeats, nRows, nCols, gamma)

# iteratively sample a reward and compare policy
N = 100
diff = dict([(s, [0] * len(s.actions)) for s in mdp.states])
entropy = dict([(s, 0) for s in mdp.states])
for i in range(N):
	s = np.random.uniform(-1, 1, (2, 1))
	theta = np.dot(L, s).T[0] + theta_bar
	pi, _ = ut.policy_iteraion(mdp, theta)
	for s in mdp.states:
		index = s.actions.index(pi[s])
		diff[s][index] += 1.0 / N
for s in mdp.states:
	E = 0
	for i in range(len(s.actions)):
		if diff[s][i] > 0:
			E -= diff[s][i] * math.log(diff[s][i])
	entropy[s] = E

# highlight high and low entropy states
entropy_sort = sorted(entropy.items(), key=operator.itemgetter(1))
s_preferences = [entropy_sort[i][0] for i in range(-1, -6, -1)]
s_constraints = [entropy_sort[i][0] for i in range(5)]

# visualize the mdp + r
reward_grid = np.zeros((nRows, nCols))
for irow in range(nRows):
	for icol in range(nCols):
		s = mdp.get_state(irow, icol)
		reward_grid[icol, irow] = mdp.R(s, [1.0, 0.5, -1.0])
fig, ax = plt.subplots()
ax.imshow(reward_grid, cmap='gray')

# show the low entropy states
for s in s_constraints:
	pos = s.position
	for index in range(len(diff[s])):
		a = s.actions[index]
		p = diff[s][index]
		plt.arrow(pos[0], pos[1], a[0]/10.0, a[1]/10.0, head_width=0.5, head_length=0.3, color=[0.0, 0.5, 0.5], alpha = min(1, p))

# show the high entropy states
for s in s_preferences:
	pos = s.position
	for index in range(len(diff[s])):
		a = s.actions[index]
		p = diff[s][index]
		plt.arrow(pos[0], pos[1], a[0]/10.0, a[1]/10.0, head_width=0.5, head_length=0.3, color=[1.0, 0.5, 0.0], alpha = min(1, p))

plt.show()
