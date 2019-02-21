'''
Utility functions needed for performing each user simulation

'''

import random
import math
import time
import numpy as np


def generate_experts():
    W = np.zeros((3, 3))
    W[:, 0] = [0.7, -0.1, -0.4]
    W[:, 1] = [0.9, 0.1, -0.42]
    W[:, 2] = [-0.1, 1.0, -0.38]
    return W


def sample_reward(mdp, D, T, eta, alpha, theta, theta_bar, A, beta):
    theta_chain = [theta]
    pi, V = policy_iteraion(mdp, theta)
    P = likelihood(mdp, theta, V, D, alpha, theta_bar, A, beta)
    watchdog, count = time.time(), 0
    while time.time() - watchdog < T:
        if random.random() < 0.9:
            theta1 = perturb_theta(theta, eta)
        else:
            theta1 = sample_theta(mdp.nFeats)
        pi1, V1 = policy_iteraion(mdp, theta1, pi)
        P1 = likelihood(mdp, theta1, V1, D, alpha, theta_bar, A, beta)
        if random.random() < min(1, P1 / P):
            theta_chain.append(theta1)
            theta, pi, P = theta1, pi1, P1
            count += 1
    print("I sampled " + str(count) + " times...")
    theta_mean = np.mean(np.array(theta_chain), axis=0).tolist()
    return theta_mean


def simulated_human(mdp, theta, alpha, H, noise):
    threshold = 1.0 - noise
    _, V = policy_iteraion(mdp, theta)
    D = {}
    for _ in range(H):
        s = mdp.sample_state()
        while s in D:
            s = mdp.sample_state()
        if random.random() < threshold:
            q = [a_likelihood(mdp, theta, V, s, a, alpha) for a in s.actions]
            P = [q[i] / sum(q) for i in range(len(q))]
            a = np.random.choice(len(q), 1, p=P)[0]
        else:
            a = random.randint(0, len(s.actions) - 1)
        D[s] = s.actions[a]
    return D


def likelihood(mdp, theta, V, D, alpha, theta_bar, A, beta):
    P = 1.0
    for s in D:
        p = a_likelihood(mdp, theta, V, s, D[s], alpha)
        Z = sum(a_likelihood(mdp, theta, V, s, a, alpha) for a in s.actions)
        P *= (p / Z)
    return P * math.exp(- beta * distance(theta, theta_bar, A))


def a_likelihood(mdp, theta, V, s, a, alpha):
    s1 = mdp.T(s, a)
    return math.exp(alpha * (mdp.R(s, theta) + mdp.gamma * V[s1]))


def distance(theta1, theta2, A):
    e = np.array(theta1) - np.array(theta2)
    return np.dot(np.dot(e.T, A), e)


def sample_theta(nFeats):
    return [random.uniform(-1.0, 1.0) for _ in range(nFeats)]


def perturb_theta(theta, eta):
    nFeats = len(theta)
    theta1 = [theta[i] for i in range(nFeats)]
    for i in range(nFeats):
        lb = max(-eta / 2.0, -1.0 - theta[i])
        ub = min(eta / 2.0, 1.0 - theta[i])
        theta1[i] += random.uniform(lb, ub)
    return theta1


def policy_iteraion(mdp, theta, pi0=None):
    if pi0:
        pi = dict([(s, pi0[s]) for s in mdp.states])
    else:
        pi = dict([(s, random.choice(s.actions)) for s in mdp.states])
    while True:
        V = policy_value(mdp, theta, pi)
        unchanged = True
        for s in mdp.states:
            max_reward = V[mdp.T(s, pi[s])]
            for a in s.actions:
                curr_reward = V[mdp.T(s, a)]
                if curr_reward > max_reward + 1e-5:
                    max_reward = curr_reward
                    pi[s] = a
                    unchanged = False
        if unchanged:
            return pi, V


def policy_value(mdp, theta, pi):
    R, T = np.zeros(mdp.nStates), np.zeros((mdp.nStates, mdp.nStates))
    EYE = np.identity(mdp.nStates)
    for s in mdp.states:
        R[mdp.get_index(s)] = mdp.R(s, theta)
        T[mdp.get_index(s), mdp.get_index(mdp.T(s, pi[s]))] = 1
    V1 = np.dot(np.linalg.inv(EYE - mdp.gamma * T), R)
    V = dict([(s, 0) for s in mdp.states])
    for s in mdp.states:
        V[s] = V1[mdp.get_index(s)]
    return V


def regret(mdp, theta_star, pi_star, pi):
    V_star = policy_value(mdp, theta_star, pi_star)
    V = policy_value(mdp, theta_star, pi)
    return sum(V_star[s] - V[s] for s in mdp.states)


def reward_error(theta_star, theta):
    return sum(abs(i[0] - i[1]) for i in zip(theta_star, theta))


class State:

    def __init__(self, position=None):
        self.position = position
        self.actions = []
        self.features = []


class GridWorld:

    def __init__(self, nFeats=3, nRows=4, nCols=4, gamma=0.9):
        self.nFeats = nFeats
        self.nRows = nRows
        self.nCols = nCols
        self.nStates = self.nRows * self.nCols
        self.gamma = gamma
        self.states = []
        for y in range(self.nRows):
            for x in range(self.nCols):
                s = State((x, y))
                s.features = [0] * nFeats
                if x == 0 and y == 1:
                    s.features = [0, 0, 1]
                elif x == 1 and y == 3:
                    s.features = [0, 1, 0]
                elif x == 3 and y == 1:
                    s.features = [1, 0, 0]
                if x > 0:
                    s.actions.append((-1, 0))
                if x < self.nCols - 1:
                    s.actions.append((1, 0))
                if y > 0:
                    s.actions.append((0, -1))
                if y < self.nRows - 1:
                    s.actions.append((0, 1))
                self.states.append(s)

    def T(self, s, a):
        return self.get_state(s.position[0] + a[0], s.position[1] + a[1])

    def R(self, state, theta):
        return sum(i[0] * i[1] for i in zip(state.features, theta))

    def get_state(self, pos_x, pos_y):
        return self.states[pos_y * (self.nCols) + pos_x]

    def get_index(self, state):
        pos = state.position
        return pos[1] * self.nCols + pos[0]

    def sample_state(self):
        return self.states[random.randint(0, self.nRows * self.nCols - 1)]
