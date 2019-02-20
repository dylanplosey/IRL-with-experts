import utils as ut
import numpy as np


# parameters
nExperts = 1000
alpha = 100
beta = 1
H = 5
T = 5
eta = 0.1
nFeats = 5
nRows = 5
nCols = 5
gamma = 0.9


# generate experts
W = ut.generate_experts(nExperts)
theta_bar = np.mean(W, axis=1)
U, S, Vt = np.linalg.svd(W)
Sinv = np.linalg.inv(np.diag(S))
A = np.dot(np.dot(U, Sinv), U.T)

# generate user
theta_star = ut.generate_experts(1).T[0]
print(theta_star)

# get world and samples
mdp = ut.GridWorld(nFeats, nRows, nCols, gamma)
D = ut.simulated_human(mdp, theta_star, alpha, H)

# BIRL
E = np.eye(nFeats)
theta_E = ut.sample_reward(mdp, D, T, eta, alpha, theta_bar, theta_bar, E, beta)
theta_A = ut.sample_reward(mdp, D, T, eta, alpha, theta_bar, theta_bar, A, beta)
print(theta_E)
print(theta_A)

print(ut.reward_error(theta_star, theta_E))
print(ut.reward_error(theta_star, theta_A))

pi_star, _ = ut.policy_iteraion(mdp, theta_star)
piE, _ = ut.policy_iteraion(mdp, theta_E, pi_star)
piA, _ = ut.policy_iteraion(mdp, theta_A, pi_star)

print(ut.regret(mdp, theta_star, pi_star, piE))
print(ut.regret(mdp, theta_star, pi_star, piA))

count = [0, 0]
for s in pi_star:
	if piE[s] != pi_star[s]:
		count[0] += 1
	if piA[s] != pi_star[s]:
		count[1] += 1
print(count)









#for _ in range(100):




"""
	theta_star = ut.sample_collaborator(theta_bar, M, beta)
	print(theta_bar)
	print(M)
	print(theta_star)



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
"""