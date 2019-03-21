import random
import numpy as np


def generate_experts():
    W = np.zeros((3, 10))
    W[:, 0] = [0.2, -0.1, -1.0]
    W[:, 1] = [0.3, 0.2, -0.9]
    W[:, 2] = [0.0, 0.0, -1.0]
    W[:, 3] = [-0.2, 0.4, -0.9]
    W[:, 4] = [-0.1, 0.3, -1.0]
    W[:, 5] = [0.4, 0.4, -0.95]
    W[:, 6] = [0.0, 0.1, -0.8]
    W[:, 7] = [0.1, 0.3, -1.0]
    W[:, 8] = [0.0, 0.5, -1.0]
    W[:, 9] = [0.5, -0.1, -1.0]
    return W


def policy_iteraion(mdp, theta, pi0=None):
    if pi0:
        pi = dict([(s, pi0[s]) for s in mdp.states])
    else:
        pi = dict([(s, random.choice(s.actions)) for s in mdp.states])
    while True:
        V = policy_value(mdp, theta, pi)
        unchanged = True
        for s in mdp.states:
            p = mdp.T(s, pi[s])
            max_reward = sum([V[index[0]] * index[1] for index in p])
            for a in s.actions:
                p1 = mdp.T(s, a)
                curr_reward = sum([V[index[0]] * index[1] for index in p1])
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
        p = mdp.T(s, pi[s])
        for index in p:
            T[mdp.get_index(s), mdp.get_index(index[0])] = index[1]
    V1 = np.dot(np.linalg.inv(EYE - mdp.gamma * T), R)
    V = dict([(s, 0) for s in mdp.states])
    for s in mdp.states:
        V[s] = V1[mdp.get_index(s)]
    return V


class State:

    def __init__(self, position=None):
        self.position = position
        self.actions = []
        self.features = []


class GridWorld:

    def __init__(self, nFeats=3, nRows=5, nCols=5, gamma=0.95):
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
                if random.random() < 0.2:
                    s.features[random.randint(0, nFeats - 1)] = 1
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
        p = []
        pos = s.position
        p.append((self.get_state(pos[0] + a[0], pos[1] + a[1]), 0.6))
        if a[0] == 0:
            if (1, 0) in s.actions and (-1, 0) in s.actions:
                p.append((self.get_state(pos[0] + 1, pos[1]), 0.2))
                p.append((self.get_state(pos[0] - 1, pos[1]), 0.2))
            elif (1, 0) in s.actions:
                p.append((self.get_state(pos[0] + 1, pos[1]), 0.4))
            elif (-1, 0) in s.actions:
                p.append((self.get_state(pos[0] - 1, pos[1]), 0.4))
        elif a[1] == 0:
            if (0, 1) in s.actions and (0, -1) in s.actions:
                p.append((self.get_state(pos[0], pos[1] + 1), 0.2))
                p.append((self.get_state(pos[0], pos[1] - 1), 0.2))
            elif (0, 1) in s.actions:
                p.append((self.get_state(pos[0], pos[1] + 1), 0.4))
            elif (0, -1) in s.actions:
                p.append((self.get_state(pos[0], pos[1] - 1), 0.4))
        return p

    def R(self, state, theta):
        return sum(i[0] * i[1] for i in zip(state.features, theta))

    def get_state(self, pos_x, pos_y):
        return self.states[pos_y * (self.nCols) + pos_x]

    def get_index(self, state):
        pos = state.position
        return pos[1] * self.nCols + pos[0]

    def sample_state(self):
        return self.states[random.randint(0, self.nRows * self.nCols - 1)]
