import numpy as np
from numpy import loadtxt

import matplotlib.pyplot as plt

algorithms = ["reinforce", "batchac","ppo", "my"]
seeds = ["1", "2", "3", "4", "5"]

mean_alg = []
std_alg = []

for alg in algorithms:
    run_data = []

    for s in seeds:
        lines = loadtxt(alg + s + ".txt", comments="#", delimiter=" ", unpack=False)
        plt.plot(lines[0], lines[1])
        plt.xlabel('Training Steps')
        plt.ylabel('Return (average of last 10,000 episodes)')
        if alg == "reinforce":
            plt.title('CartPole-v1: Reinforce')
        elif alg == "batchac":
            plt.title('CartPole-v1: Batch Actor-Critic')
        elif alg == "ppo":
            plt.title('CartPole-v1: PPO')
        elif alg == "my":
            plt.title('CartPole-v1: my Algorithm (PPO w/o clip)')

        run_data.append(lines[1])
    plt.ylim([0, 500])
    plt.legend(seeds, title="Run", loc='lower right')
    plt.savefig(alg+".png")
    plt.close()

    mean_alg.append(np.mean(run_data, axis=0))
    std_alg.append(np.std(run_data, axis=0))

for j in range(len(algorithms)):
    plt.plot(lines[0], mean_alg[j])
    plt.fill_between(lines[0], mean_alg[j] - std_alg[j], mean_alg[j] + std_alg[j], alpha=0.2)

plt.title("CartPole-v1: Comparing Policy Gradient Methods (averaged over 5 runs)")
plt.xlabel('Training Steps')
plt.ylabel('Return (average of last 10,000 episodes)')
plt.legend(algorithms, title="Algorithm", loc='lower right')
plt.savefig("fig4.png")
plt.show()
