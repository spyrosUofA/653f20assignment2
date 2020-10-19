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

    # Learner setup block
    torch.manual_seed(seed)

    ####### Start
    # Actor & Critic networks
    actor = nn.Sequential(
        nn.Linear(o_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, a_dim),
        nn.Softmax(dim=-1))

    critic = nn.Sequential(
        nn.Linear(o_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1))

    # Actor & Critic optimizers
    opt_act = torch.optim.Adam(actor.parameters(), lr=0.005)
    opt_cri = torch.optim.Adam(critic.parameters(), lr=0.005)

    # Actionspace is [0, 1] in cartpole
    action_space = np.arange(env.action_space.n)

    # PPO parameters
    epsilon = 0.2
    gamma = 1
    lam = 1

    # Batch update parameters
    k = 1   # initial episode number
    K = 200 # nb of episodes per batch
    E = 5   # Nb of epochs
    N = 6   # Nb of mini-batches
    M = N   # Nb of mini-batches used

    # Lambda return for an episode
    def lambda_return(lam, gamma, s_vec, r_vec, a_critic):
        # R_vec = [R1, R2, ..., R_T]
        # S_vec = [s0, s1, ...., S_(T-1)]
        G_l = []
        T = len(r_vec) - 1

        G_l.append(r_vec[-1]) #G^l_(T-1) = R_T

        while T > 0:
            G_l.append(gamma*(1-lam)*a_critic(torch.FloatTensor(s_vec[T-1])).squeeze().detach().numpy() + r_vec[T] + gamma*lam*G_l[-1])
            T -= 1

        return G_l[::-1]

    # State, Action, Reward, Return (to be emptied after each episode)
    s_epi = []
    a_epi = []
    r_epi = []
    g_epi = []

    # State, Action, Reward, Return (to be emptied after each batch)
    s_batch = []
    a_batch = []
    r_batch = []
    g_batch = []
    ####### End

    # Experiment block starts
    ret = 0
    rets = []
    avgrets = []
    o = env.reset()
    num_steps = 500000
    checkpoint = 10000
    for steps in range(num_steps):

        #print(steps)

        # Select an action
        ####### Start
        a = np.random.choice(a=action_space, p=actor(torch.FloatTensor(o)).detach().numpy())
        ####### End

        # Observe
        op, r, done, infos = env.step(a)

        # Learn
        ####### Start
        # Here goes your learning update
        s_epi.append(o)
        a_epi.append(a)
        r_epi.append(r)

        if done:

            # Episode Lambda returns
            g_epi = lambda_return(lam, gamma, s_epi, r_epi, critic)

            # Adding episode to batch
            s_batch.extend(s_epi)
            a_batch.extend(a_epi)
            g_batch.extend(g_epi)

            # Clear after each episode
            s_epi = []
            a_epi = []
            r_epi = []
            g_epi = []

            # If episode is over and B episodes added to batch, then learn
            if k % K == 0:
                s_batch = torch.FloatTensor(s_batch)
                a_batch = torch.LongTensor(a_batch)
                g_batch = torch.FloatTensor(g_batch).squeeze()
                h_batch_prime = g_batch - critic(torch.FloatTensor(s_batch)).squeeze()  # take grad on this (it is now like a vector)

                h_batch_prime = (h_batch_prime - h_batch_prime.mean()) / h_batch_prime.std()  # normalizing
                h_batch_old = h_batch_prime.detach()  # do not take grad on this

                policy_prime = torch.gather(actor(torch.FloatTensor(s_batch)), 1, a_batch.unsqueeze(1)).squeeze()
                policy_old = policy_prime.detach()

               # print(policy_prime)

                for epoch in range(E):
                    # Shuffle
                    indx = np.random.permutation(a_batch.size()[0])

                    # Nb of timesteps in each batch
                    mini_batch_size = int(np.ceil(len(indx)/N))

                    # indices of each batch
                    mini_batch_indx = [indx[x:(x + mini_batch_size)] for x in range(0, len(indx), mini_batch_size)]

                    for mini in range(M):
                        #print(np.amin(mini_batch_indx))
                        shuffled_states = s_batch[mini_batch_indx[mini]]
                        shuffled_actions = a_batch[mini_batch_indx[mini]]

                        shuffled_policy_old = policy_old[mini_batch_indx[mini]]
                        shuffled_h_old = h_batch_old[mini_batch_indx[mini]]

                        shuffled_policy = torch.gather(actor(torch.FloatTensor(shuffled_states)), 1, shuffled_actions.unsqueeze(1)).squeeze()
                        shuffled_h = g_batch[mini_batch_indx[mini]] - critic(torch.FloatTensor(shuffled_states)).unsqueeze(1) # check thjis unsqeueeze

                        # update actor....
                        loss_act = -torch.mean(shuffled_policy / shuffled_policy_old * shuffled_h_old)
                        opt_act.zero_grad()
                        loss_act.backward()
                        opt_act.step()

                        # update critic...
                        loss_crit = (shuffled_h ** 2).mean()
                        opt_cri.zero_grad()
                        loss_crit.backward()
                        opt_cri.step()

                # Emptying after each batch
                s_batch = []
                a_batch = []
                #r_batch = []
                g_batch = []
                #h_batch_prime = []
                #h_batch = []

            # move onto next episode
            k += 1

        # Learning ends

        # Update environment
        o = op

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
    #plt.show()


if __name__ == "__main__":
    main()
