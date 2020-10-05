import sys
import numpy as np
import matplotlib as mpl
mpl.use("TKAgg")
import matplotlib.pyplot as plt
import gym
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
  seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

  # Task setup block starts
  # Do not change
  env = gym.make('CartPole-v1')
  env.seed(seed)
  o_dim = env.observation_space.shape[0]
  a_dim = env.action_space.n
  # Task setup block end

  # Learner setup block
  torch.manual_seed(seed)
  ####### Start
  ####### End

  # Experiment block starts
  ret = 0
  rets = []
  avgrets = []
  o = env.reset()
  num_steps = 500000
  checkpoint = 10000
  for steps in range(num_steps):

    # Select an action
    ####### Start
    # Replace the following statement with your own code for
    # selecting an action
    a = np.random.randint(a_dim)
    ####### End

    # Observe
    op, r, done, infos = env.step(a)

    # Learn
    ####### Start
    # Here goes your learning update
    ####### End

    # Log
    ret += r
    if done:
      rets.append(ret)
      ret = 0
      o = env.reset()

    if (steps+1) % checkpoint == 0:
      avgrets.append(np.mean(rets))
      rets = []
      plt.clf()
      plt.plot(range(checkpoint, (steps+1)+checkpoint, checkpoint), avgrets)
      plt.pause(0.001)
  name = sys.argv[0].split('.')[-2].split('_')[-1]
  data = np.zeros((2, len(avgrets)))
  data[0] = range(checkpoint, num_steps+1, checkpoint)
  data[1] = avgrets
  np.savetxt(name+str(seed)+".txt", data)
  plt.show()


if __name__ == "__main__":
  main()
