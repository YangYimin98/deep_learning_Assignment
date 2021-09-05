from datetime import datetime

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
import torch.nn as nn
# from torch.autograd import Variable

from read_file import *

"""
Reference:

Conv_1 groups: https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
"""


class ModelCNN(nn.Module):
    def __init__(self, input_channel, input_len, latent_channel, k, s, p, pool, dropout=0.5, num_classes=4):
        self.input_channel = input_channel
        self.input_len = input_len
        self.latent_channel = latent_channel

        super(ModelCNN, self).__init__()
        self.linear1 = nn.Linear(input_channel, latent_channel)
        # self.conv1 = nn.Conv1d(latent_channel, latent_channel, kernel_size=k, stride=s, padding=p, groups=latent_channel)  # temporal conv that each kernal only produce one channel output
        self.conv1 = nn.Conv1d(latent_channel, latent_channel, kernel_size=k, stride=s, padding=p)  # temporal conv that each kernal only produce one channel output
        self.relu = nn.ReLU()
        self.mp = nn.AdaptiveMaxPool1d(np.int(input_len / pool))
        # self.mp = nn.AdaptiveMaxPool1d(np.int(input_len / pool))
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(np.int(input_len / pool) * latent_channel, num_classes)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.conv1(out.permute(0, 2, 1))
        out = self.relu(out)
        out = self.mp(out)
        out = self.dropout(out)
        out = self.linear2(out.reshape(out.shape[0], -1))
        out = self.soft(out)
        return out

    def training_model(self, x, y, epochs, batch_size, lr, l1_weight=0):
        self.train()
        print(x.shape, y.shape, epochs, batch_size, lr)
        criterion = nn.NLLLoss()  # MSE
        batches = x.shape[0]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Start training
        loss_log = np.zeros((epochs, 1))

        for e_index in range(epochs):
            for batch in range(batches):
                outputs = self.forward(x[batch, :, :, :])
                optimizer.zero_grad()
                if l1_weight:
                    regularization_loss = 0
                    for param in self.parameters():
                        regularization_loss += torch.sum(torch.abs(param))
                    loss = criterion(torch.log(outputs), y[batch, :].reshape(batch_size)) + l1_weight * regularization_loss
                else:
                    loss = criterion(torch.log(outputs), y[batch, :].reshape(batch_size))
                loss.backward()
                optimizer.step()
                loss_log[e_index, 0] = loss.item()
            if e_index % 10 == 0:
                print(datetime.now(), "Epoch: {0}, Training loss: {1:1.8f}".format(e_index, loss.item()))
        # Save model
        # torch.save(self.modules(), 'LSTM.pkl')
        self.eval()
        return loss_log




