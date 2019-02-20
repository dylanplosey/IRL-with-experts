import utils as ut
import numpy as np


# parameters
nExperts = 1000
alpha = 10
noise = 0.4
beta = 100
H = 5
T = 1
eta = 0.1
nFeats = 5
FeatsInc = [1]*5
nRows = 5
nCols = 5
gamma = 0.9


# generate experts
W = ut.generate_experts(nExperts)
theta_bar = np.mean(W, axis=1)
U, S, Vt = np.linalg.svd(W)
Sinv = np.linalg.inv(np.diag(S))
A = np.dot(np.dot(U, Sinv), U.T)

regret = [0, 0]
regret_bad = [0, 0]
regret_good = [0, 0]

for _ in range(1000):

	# generate user
	theta_star = ut.generate_experts(1).T[0]

	# get world and samples
	mdp = ut.GridWorld(nFeats, nRows, nCols, gamma, FeatsInc)
	D = ut.simulated_human(mdp, theta_star, alpha, H, noise)

	# BIRL
	E = np.diag([1]*5)
	theta_0 = ut.sample_theta(nFeats)
	theta_E = ut.sample_reward(mdp, D, T, eta, alpha, theta_0, theta_bar, E, beta)
	theta_A = ut.sample_reward(mdp, D, T, eta, alpha, theta_0, theta_bar, A, beta)

	pi_star, _ = ut.policy_iteraion(mdp, theta_star)
	piE, _ = ut.policy_iteraion(mdp, theta_E, pi_star)
	piA, _ = ut.policy_iteraion(mdp, theta_A, pi_star)

	rE = ut.regret(mdp, theta_star, pi_star, piE)
	rA = ut.regret(mdp, theta_star, pi_star, piA)
	print(rE, rA)

	z = max([rE, rA, 1e-5])
	regret[0] += rE/z
	regret[1] += rA/z
	print(regret)

	# Test in Bad Worlds
	mdp_bad = ut.GridWorld(nFeats, nRows, nCols, gamma, [0,0,1,1,1])
	pi_star, _ = ut.policy_iteraion(mdp_bad, theta_star)
	piE, _ = ut.policy_iteraion(mdp_bad, theta_E, pi_star)
	piA, _ = ut.policy_iteraion(mdp_bad, theta_A, pi_star)

	rE_b = ut.regret(mdp_bad, theta_star, pi_star, piE)
	rA_b = ut.regret(mdp_bad, theta_star, pi_star, piA)

	z = max([rE_b, rA_b, 1e-5])
	regret_bad[0] += rE_b/z
	regret_bad[1] += rA_b/z
	print(regret_bad)

	# Test in Good Worlds
	mdp_good = ut.GridWorld(nFeats, nRows, nCols, gamma, [1,1,1,0,0])
	pi_star, _ = ut.policy_iteraion(mdp_good, theta_star)
	piE, _ = ut.policy_iteraion(mdp_good, theta_E, pi_star)
	piA, _ = ut.policy_iteraion(mdp_good, theta_A, pi_star)

	rE_g = ut.regret(mdp_good, theta_star, pi_star, piE)
	rA_g = ut.regret(mdp_good, theta_star, pi_star, piA)

	z = max([rE_g, rA_g, 1e-5])
	regret_good[0] += rE_g/z
	regret_good[1] += rA_g/z
	print(regret_good)



'''

what are the right dependent variables?

1. regret across the demonstrated MDP
2. regret in MDP with only the "preferred" states
3. regret in MDP with only the "constrained" states


what are the right independent variables?

1. alpha (so we can have bad demonstrations)


'''