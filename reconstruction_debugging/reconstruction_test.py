import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys


class ReconstructionModel(nn.Module):
    def __init__(self, state_shape, observation_shape, kernel_size=5):
        super(ReconstructionModel, self).__init__()
        
        self.state_shape = state_shape
        self.observation_shape = observation_shape

        hidden_size = 64

        layers = []
        layers.append(nn.ConvTranspose2d(1, out_channels=64, kernel_size=5, stride=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(64, out_channels=32, kernel_size=5, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(32, 3, kernel_size=13, stride=2))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        output = self.model(inputs)
        output = output[:, :, :64, :64]
        return output

def main():
    epochs = 100
    model = ReconstructionModel()
    optimizer = torch.optim.Adam(lr=5e-4, model.parameters())

    data = torchvision.datasets.ImageFolder(root="reconstruction_debugging/training_samples")
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=100, shuffle=True, num_workers=8)

    for e in epochs:
        for latent, images in data_loader:
            reconstructions = model(latent)
            loss = F.l1_loss(reconstructions, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
