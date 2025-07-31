import numpy as np

class Bandit:
    
    def __init__(self, k):
        self.k = k
        self.means = np.random.rand(k)
        self.N_greedy = np.zeros(k)
        self.N_UCB = np.zeros(k)

    def first_round(self):
        self.N_greedy += np.ones(self.k)
        self.N_UCB += np.ones(self.k)
        return np.array([self.gen_reward(i) for i in range(self.k)])

    def gen_reward(self, a):
        # return np.random.normal(self.means[a], 2)
        return np.random.binomial(2, self.means[a])

    def UCB_Q_up(self, Q, r, a):
        return Q[a] + (r - Q[a]) / self.N_UCB[a]
    
    def greedy_Q_up(self, Q, r, a):
        return Q[a] + (r - Q[a]) / self.N_greedy[a]

    def greedy_as(self, Q):
        a = np.argmax(Q)
        self.N_greedy[a] += 1
        return a

    def UCB_as(self, Q, t):
        a = np.argmax(Q + np.sqrt(2 * np.log(t)/self.N_UCB))
        self.N_UCB[a] += 1
        return a
