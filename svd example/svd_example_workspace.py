import svd_example_utils as ut
import matplotlib.pyplot as plt
import numpy as np

# parameters
alpha = 1
beta = 3

# collect expert weights offline
W = ut.generate_experts()

# find latent representation
theta_bar = np.mean(W, axis=1)
W_zero = W - np.asarray([theta_bar] * 5).T
U, S, Vt = np.linalg.svd(W_zero)
Sm = np.diag(S)
L = np.dot(U, Sm[:, [0, 1]])

# get grid world
nFeats = 3
nRows = 4
nCols = 4
gamma = 0.9
mdp = ut.GridWorld(nFeats, nRows, nCols, gamma)

# choose a = (-1, 0) for "bad" demonstration or (0, 1) for "good" demonstration
a = (0, 1)
D = [mdp.get_state(1, 1), a]
theta_star = [0.1, 0.2, -1.0]
print("The true weights are: " + str(theta_star))

# BIRL : uniform, baseline, svd
theta_0, T, eta = theta_bar, 5, 0.1
theta_U = ut.sample(mdp, D, T, eta, alpha, theta_0, theta_bar, L, 0.0, False)
print("The weights I learned with a uniform prior are: " + str(theta_U))
theta_E = ut.sample(mdp, D, T, eta, alpha, theta_0, theta_bar, L, beta, False)
print("The weights I learned with a L1 norm prior are: " + str(theta_E))
theta_L = ut.sample(mdp, D, T, eta, alpha, theta_0, theta_bar, L, beta, True)
print("The weights I learned with a SVD prior are: " + str(theta_L))

# learned policies
pi_star, V_star = ut.policy_iteraion(mdp, theta_star)
pi_uniform, _ = ut.policy_iteraion(mdp, theta_U)
pi_baseline, _ = ut.policy_iteraion(mdp, theta_E)
pi_svd, _ = ut.policy_iteraion(mdp, theta_L)

# print the regret
V_uniform = ut.policy_value(mdp, theta_star, pi_uniform)
V_baseline = ut.policy_value(mdp, theta_star, pi_baseline)
V_svd = ut.policy_value(mdp, theta_star, pi_svd)
r_uniform = ut.regret(V_uniform, V_star)
r_baseline = ut.regret(V_baseline, V_star)
r_svd = ut.regret(V_svd, V_star)
print("My regret with a uniform prior is: " + str(r_uniform))
print("My regret with a L1 norm prior is: " + str(r_baseline))
print("My regret with a SVD prior is: " + str(r_svd))

# identify the worst action
loss_uniform, s_uniform = ut.state_regret(V_uniform, V_star)
loss_baseline, s_baseline = ut.state_regret(V_baseline, V_star)
loss_svd, s_svd = ut.state_regret(V_svd, V_star)
print("Uniform prior: my worst action loses " + str(loss_uniform) + " at state " + str(s_uniform.position))
print("L1 norm prior: my worst action loses " + str(loss_baseline) + " at state " + str(s_baseline.position))
print("SVD prior: my worst action loses " + str(loss_svd) + " at state " + str(s_svd.position))

# visualize the mdp + r
reward_grid = np.zeros((nRows, nCols))
for irow in range(nRows):
	for icol in range(nCols):
		s = mdp.get_state(irow, icol)
		reward_grid[icol, irow] = mdp.R(s, [0.5, 1.0, -1.0])
fig, ax = plt.subplots()
ax.imshow(reward_grid, cmap='gray')

# plot the policy learned by each agent
for s in mdp.states:
	pos, a = s.position, pi_uniform[s]
	plt.arrow(pos[0], pos[1], a[0]/10.0, a[1]/10.0, head_width=0.5, head_length=0.2, color=[0.75, 0.75, 0.75])
	a = pi_baseline[s]
	plt.arrow(pos[0], pos[1], a[0]/10.0, a[1]/10.0, head_width=0.5, head_length=0.2, color=[0.3, 0.3, 0.3])
	a = pi_svd[s]
	plt.arrow(pos[0], pos[1], a[0]/10.0, a[1]/10.0, head_width=0.5, head_length=0.2, color=[1.0, 0.5, 0.0])

plt.show()
