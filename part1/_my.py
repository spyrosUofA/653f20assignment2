import numpy as np
import matplotlib as mpl

mpl.use("TKAgg")
import matplotlib.pyplot as plt
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Task setup block starts
    # Do not change
    torch.manual_seed(1000)
    dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    # Task setup block end

    # Learner setup block
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])
    torch.manual_seed(seed)  # do not change. This is for learners randomization

    model = nn.Sequential(nn.Linear(1*28*28, 150),
                          nn.ReLU(),
                          nn.Linear(150, 50),
                          nn.ReLU(),
                          nn.Linear(50, 10),
                          nn.LogSoftmax(dim=-1))

    neg_logli_loss = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Params for batch learning
    T = 2000  # Number of times steps
    C = T  # Buffer Capacity
    b = []  # Buffer
    E = 10  # Nb of epochs
    N = 5  # Nb of mini-batches
    M = N  # Nb of mini-batches used

    lam = 0.5
    alpha = 1e-4

    # Experiment block starts
    errors = []
    checkpoint = 1000
    correct_pred = 0
    for idx, (image, label) in enumerate(loader):

        # Observe
        label = label.to(device=device)
        image = image.to(device=device)

        # Make a prediction of label
        image = image.view(image.shape[0], -1)
        prob_label = model(image)
        pred_label = torch.argmax(prob_label)

        # Evaluation
        correct_pred += (pred_label == label).sum()

        # learning update

        # add to buffer if capacity permits
        if len(b) < C:
            b.append([image, label])

        # if low confidence on current sample, do update immediately
        if torch.max(prob_label) < 0.25: #min(lam*(1+alpha)**idx, 0.95):
            # if batch is big enough,  do batch update with epochs
            if len(b) > N*E:
                for epoch in range(2*E):
                    # Shuffle samples
                    np.random.shuffle(b)

                    # Split into N mini-batches
                    mini_batch_size = int(C / N) + (C % N > 0)
                    mini_batch = [b[x:(x + mini_batch_size)] for x in range(0, len(b), mini_batch_size)]

                    # Update theta based on M of the mini-batches
                    for mini in range(M):
                        # separating images and labels from mini-batches
                        images = [il_tuple[0] for il_tuple in mini_batch[mini]]
                        labels = [il_tuple[1] for il_tuple in mini_batch[mini]]

                        # formatting to tensors
                        images = torch.stack(images).to(device)
                        labels = torch.from_numpy(np.array(labels))

                        # prediction
                        prob_labels = model(images.squeeze(1))

                        # Update theta using backprop
                        optimizer.zero_grad()
                        loss = neg_logli_loss(prob_labels, labels)
                        loss.backward()
                        optimizer.step()
            else:
                # splitting up b into images and labels
                images = [il_tuple[0] for il_tuple in b]
                labels = [il_tuple[1] for il_tuple in b]

                # formatting to tensors
                images = torch.stack(images).to(device)
                labels = torch.from_numpy(np.array(labels))

                # prediction
                prob_labels = model(images.squeeze(1))

                # Update theta using backprop
                optimizer.zero_grad()
                loss = neg_logli_loss(prob_labels, labels)
                loss.backward()
                optimizer.step()

            # empty buffer
            b = []

        # if we haven't had any low confidence, then update when buffer is full
        elif len(b) == C:
            for epoch in range(E):
                # Shuffle samples
                np.random.shuffle(b)

                # Split into N mini-batches
                mini_batch_size = int(C / N) + (C % N > 0)
                mini_batch = [b[x:(x + mini_batch_size)] for x in range(0, len(b), mini_batch_size)]

                # Update theta based on M of the mini-batches
                for mini in range(M):
                    # separating images and labels from mini-batches
                    images = [il_tuple[0] for il_tuple in mini_batch[mini]]
                    labels = [il_tuple[1] for il_tuple in mini_batch[mini]]

                    # formatting
                    images = torch.stack(images).to(device)
                    labels = torch.from_numpy(np.array(labels))

                    # prediction
                    prob_labels = model(images.squeeze(1))

                    # Update theta using backprop
                    optimizer.zero_grad()
                    loss = neg_logli_loss(prob_labels, labels)
                    loss.backward()
                    optimizer.step()

            # clear buffer
            b = []

        # Log
        if (idx + 1) % checkpoint == 0:
            error = float(correct_pred) / float(checkpoint) * 100
            print(error)
            errors.append(error)
            correct_pred = 0

            plt.clf()
            plt.plot(range(checkpoint, (idx + 1) + checkpoint, checkpoint), errors)
            plt.ylim([0, 100])
            plt.pause(0.001)
    name = sys.argv[0].split('.')[-2].split('_')[-1]
    data = np.zeros((2, len(errors)))
    data[0] = range(checkpoint, 60000 + 1, checkpoint)
    data[1] = errors
    np.savetxt(name + str(seed) + ".txt", data)
    #plt.show()


if __name__ == "__main__":
    main()
