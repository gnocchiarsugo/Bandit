import numpy as np
import matplotlib.pyplot as plt

# --- Setup ---
np.random.seed(42)

# Simulate a bandit problem
n_arms = 5
n_rounds = 5000
true_means = np.random.rand(n_arms)  # true reward probabilities of each arm

# --- Greedy Algorithm ---
def greedy_bandit(n_rounds, true_means):
    n_arms = len(true_means)
    counts = np.zeros(n_arms)
    rewards = np.zeros(n_arms)
    total_rewards = []

    for t in range(n_rounds):
        if t < n_arms:
            arm = t  # try each arm once
        else:
            arm = np.argmax(rewards / (counts + 1e-5))  # avoid division by zero

        reward = np.random.normal(true_means[arm], 1)
        # reward = np.random.binomial(1, true_means[arm])
        counts[arm] += 1
        rewards[arm] += reward
        total_rewards.append(reward)

    return np.cumsum(total_rewards)

# --- UCB1 Algorithm ---
def ucb_bandit(n_rounds, true_means):
    n_arms = len(true_means)
    counts = np.zeros(n_arms)
    rewards = np.zeros(n_arms)
    total_rewards = []

    for t in range(n_rounds):
        if t < n_arms:
            arm = t  # play each arm once
        else:
            ucb_values = (rewards / (counts + 1e-5)) + np.sqrt(2 * np.log(t + 1) / (counts + 1e-5))
            arm = np.argmax(ucb_values)

        # reward = np.random.binomial(1, true_means[arm])
        reward = np.random.normal(true_means[arm], 1)
        counts[arm] += 1
        rewards[arm] += reward
        total_rewards.append(reward)

    return np.cumsum(total_rewards)

# --- Run both algorithms ---
greedy_cum_rewards = greedy_bandit(n_rounds, true_means)
ucb_cum_rewards = ucb_bandit(n_rounds, true_means)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(greedy_cum_rewards, label='Greedy')
plt.plot(ucb_cum_rewards, label='UCB1')
plt.title("Cumulative Reward: Greedy vs UCB")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
