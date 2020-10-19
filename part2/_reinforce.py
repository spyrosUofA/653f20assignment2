import sys
import numpy as np
import matplotlib as mpl

mpl.use("TKAgg")
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn

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


    # Learner setup block starts
    torch.manual_seed(seed)

    ####### Start
    # policy estimate
    policy = nn.Sequential(
         nn.Linear(o_dim, 32),
         nn.ReLU(),
         nn.Linear(32, 16),
         nn.ReLU(),
         nn.Linear(16, a_dim),
         nn.Softmax(dim=-1))

    # optimizer for policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    # cartpole action space
    action_space = np.arange(env.action_space.n)

    # batch params
    k = 1  # episode number
    K = 20  # nb episodes in each batch

    # Used to calculate Loss
    R_epi = []
    G_epi = []
    G_batch = []
    sample_pol = []
    

    # Learner setup block ends

    # Experiment block starts
    ret = 0
    rets = []
    avgrets = []
    o = env.reset()
    num_steps = 500000
    checkpoint = 10000
    for steps in range(num_steps):

        # Select an action
        a = np.random.choice(a = action_space, p = policy(torch.FloatTensor(o)).detach().numpy())

        # Observe
        op, r, done, infos = env.step(a)

        # Learn
        ####### Start
        # recording R_t+1 and pi(a_t|s_t, theta_k)
        R_epi.append(r)
        sample_pol.append(policy(torch.FloatTensor(o))[a])

        if done:
            # Episode Returns [G_0, ..., G_(T-1)]
            G_epi = np.cumsum(R_epi[::-1])[::-1]

            # Adding returns to batch
            G_batch.extend(G_epi)

            # Clear after each episode
            R_epi = []
            #G_epi = []
            #print(G_epi[0])
            
            # If K episodes have completed, then time to learn
            if k % K == 0:

                # Formatting to tensor
                sample_pol = torch.stack(sample_pol).to(device)
                G_batch = torch.FloatTensor(G_batch).detach()

                # Defining loss
                loss = -(torch.log(sample_pol) * G_batch).mean()

                # update network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # empty everything
                sample_pol = []
                G_batch = []

            # move onto next episode
            k += 1

        # update state
        o = op
        ####### End

        # Log
        ret += r
        if done:
            rets.append(ret)
            ret = 0
            o = env.reset()

        if (steps + 1) % checkpoint == 0:
            avgrets.append(np.mean(rets))
            rets = []
            plt.clf()
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            plt.pause(0.001)
    name = sys.argv[0].split('.')[-2].split('_')[-1]
    data = np.zeros((2, len(avgrets)))
    data[0] = range(checkpoint, num_steps + 1, checkpoint)
    data[1] = avgrets
    np.savetxt(name + str(seed) + ".txt", data)
    # plt.show()

if __name__ == "__main__":
    main()
