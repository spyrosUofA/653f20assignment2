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

    # Neural Network w/ logSoftmax
    model = nn.Sequential(nn.Linear(1*28*28, 128),
                          nn.ReLU(),
                          nn.Linear(128, 128),
                          nn.ReLU(),
                          nn.Linear(128, 10),
                          nn.LogSoftmax(dim=1))

    # negative log likelihood loss as loss
    neg_logli_loss = nn.NLLLoss()

    # adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Experiment block starts
    errors = []
    checkpoint = 1000
    correct_pred = 0
    for idx, (image, label) in enumerate(loader):

        # Observe
        label = label.to(device=device)
        image = image.to(device=device)

        # Make a prediction of label
        prob_label = model(image.view(image.shape[0], -1))
        pred_label = torch.argmax(prob_label)

        # Evaluation
        correct_pred += (pred_label == label).sum()

        # learning update
        if torch.max(prob_label) < 0.75:
            optimizer.zero_grad()
            loss = neg_logli_loss(prob_label, label)
            loss.backward()
            optimizer.step()

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
