import numpy as np


class Bandit:
    """
    Rewards are extracted from normal distribution with random mean and var = 1
    """
    def __init__(self, k):
        """
        :param k: number of arms
        """
        self.k = k
        # Means of reward distribution
        self.means = np.random.rand(k)
        # Number of times arm N[i] is selected
        self.N = np.zeros(k)

    def first_round(self):
        """
        Test all arms
        :return: Q_0
        """
        self.N += np.ones(self.N.size)
        return np.array([self.gen_reward(i) for i in range(self.N.size)])

    def gen_reward(self, a):
        """
        :param a: Generate reward for action a
        :return: Return reward
        """
        return np.random.normal(self.means[a], 1)

    def greedyQ_up(self, Q, r, a):
        return Q[a] + (r - Q[a]) / self.N[a]

    def greedy_as(self, Q):
        a = np.argmax(Q)
        self.N[a] += 1
        return a, Q[a]

    def UCB_as(self, Q, alpha, t):
        a = np.argmax(Q + np.sqrt(alpha * np.log(t)/(2 * self.N)))
        return a, Q[a]
