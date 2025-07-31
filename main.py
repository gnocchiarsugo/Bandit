import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from Bandit import *

if __name__ == '__main__':

    # Run and problem patameters
    T = 10000
    runs = 200
    k = 5

    Q_greedy = np.zeros(k)
    Q_UCB = np.zeros(k)

    # Regret
    L_greedy = np.ones(T)
    L_UCB = np.ones(T)

    # Optimal action percentage
    P_greedy = np.zeros(T)
    P_UCB = np.zeros(T)

    opt_cnt_greedy = 0
    opt_cnt_UCB = 0

    for j in tqdm(range(runs)):
        bandit = Bandit(k)
        opt_index = np.argmax(bandit.means)
        opt_v = bandit.means[opt_index]
        Q_greedy = bandit.first_round()
        Q_UCB = Q_greedy
        for i in range(T):
            # Action selection
            a_greedy = bandit.greedy_as(Q_greedy)
            a_UCB = bandit.UCB_as(Q_UCB, i + 1)

            if a_greedy == opt_index:
                opt_cnt_greedy += 1
            if a_UCB == opt_index:
                opt_cnt_UCB += 1

            # Reward distribution
            r_greedy = bandit.gen_reward(a_greedy)
            r_UCB = bandit.gen_reward(a_UCB)

            Q_greedy[a_greedy] = bandit.greedy_Q_up(Q_greedy, r_greedy, a_greedy)
            Q_UCB[a_UCB] = bandit.UCB_Q_up(Q_UCB, r_UCB, a_UCB)

            L_greedy[i] += (opt_v - Q_greedy[a_greedy] - L_greedy[i])/(j + 1)
            L_UCB[i] += (opt_v - Q_UCB[a_UCB] - L_UCB[i]) / (j + 1)

            P_greedy[i] += (opt_cnt_greedy / (i + 1) - P_greedy[i]) / (j + 1)
            P_UCB[i] += (opt_cnt_UCB / (i + 1) - P_UCB[i]) / (j + 1)

        opt_cnt_greedy = 0
        opt_cnt_UCB = 0

    plt.plot(np.arange(T), P_greedy)
    plt.plot(np.arange(T), P_UCB)
    plt.show()
    # np.savetxt('R.txt', R, fmt='%1.9e', delimiter=',', newline='],\n[')
    # np.savetxt('L.txt', L, fmt='%1.9e', delimiter=',', newline='],\n[')
    # plt.plot(np.arange(P.size), P)
    plt.plot(np.arange(T), L_greedy)
    plt.plot(np.arange(T), L_UCB)
    plt.show()
