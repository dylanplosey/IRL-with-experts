import utils as ut
import numpy as np


# parameters
alpha = 5
beta = 10
H = 5
T = 5
eta = 0.1
n_samples = 100
nFeats = 4
nRows = 5
nCols = 5
gamma = 0.9


regret_total = [0] * 3

for _ in range(10):

	# generate experts and user
	M, theta_bar = ut.generate_experts(nFeats)
	theta_star = ut.sample_collaborator(theta_bar, M, beta, n_samples, eta)

	# get world and samples
	mdp = ut.GridWorld(nFeats, nRows, nCols, gamma)
	D = ut.simulated_human(mdp, theta_star, alpha, H)

	# BIRL with Vanilla, Euclidean Norm, Proposed
	theta = []
	E = np.diag([sum(np.diag(M)) / 4] * 4)
	theta.append(ut.sample_reward(mdp, D, T, eta, alpha, theta_bar, theta_bar, E, 0))
	theta.append(ut.sample_reward(mdp, D, T, eta, alpha, theta_bar, theta_bar, E, beta))
	theta.append(ut.sample_reward(mdp, D, T, eta, alpha, theta_bar, theta_bar, M, beta))

	# normalized regret
	regret = [ut.regret(mdp, theta_star, theta[i]) for i in range(len(theta))]
	Z = max(regret)
	if Z == 0.0:
		Z = 1.0
	regret_total = [regret_total[i] + regret[i] / Z for i in range(len(theta))]

	print(regret_total)
