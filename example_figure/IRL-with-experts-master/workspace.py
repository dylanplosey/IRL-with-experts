import utils as ut
import numpy as np

# parameters
alpha = 5
beta = 1
T = 30
eta = 0.1
nFeats = 3

# generate experts
W = ut.generate_experts()
theta_bar = np.mean(W, axis=1)
U, S, Vt = np.linalg.svd(W)
Sinv = np.linalg.inv(np.diag(S))
A = np.dot(U, np.dot(Sinv, U.T))

# get world and samples
mdp = ut.GridWorld()
D = {mdp.get_state(1, 1): (-1, 0)}

# BIRL
E = np.diag([np.min(A.diagonal())] * nFeats)
theta_0 = theta_bar
theta_U = ut.sample_reward(mdp, D, T, eta, alpha, theta_0, theta_bar, E, 0)
theta_E = ut.sample_reward(mdp, D, T, eta, alpha, theta_0, theta_bar, E, beta)
theta_A = ut.sample_reward(mdp, D, T, eta, alpha, theta_0, theta_bar, A, beta)

# results
print(theta_U)
print(theta_E)
print(theta_A)
