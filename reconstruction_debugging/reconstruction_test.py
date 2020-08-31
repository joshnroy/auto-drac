import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import cv2
from tqdm import trange

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ReconstructionModel(nn.Module):
    def __init__(self):
        super(ReconstructionModel, self).__init__()
        
        layers = []
        layers.append(nn.ConvTranspose2d(1, out_channels=128, kernel_size=3, stride=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(128, out_channels=64, kernel_size=3, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2))
        layers.append(nn.Sigmoid())

        self.conv_model = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU(inplace=True))

        self.fc_model = nn.Sequential(*layers)

        self.state_shape = (16, 16)

    def forward(self, inputs):
        output = self.fc_model(inputs)
        output = torch.reshape(output, (-1, 1) + self.state_shape)
        output = self.conv_model(output)
        output = output[:, :, :64, :64] * 255.
        return output

def main():
    epochs = 1000
    batch_size = 128

    if False:
        print("Loading Data")
        data = np.load("reconstruction_debugging/training_samples/training_samples.npz", allow_pickle=True)["data"]
        print("Creating Model")
        data = list(zip(*data))
        np.savez_compressed("reconstruction_debugging/training_samples/separated_training_samples", images=np.transpose(data[1], (0, 3, 1, 2)), latents=data[0])
        sys.exit()
    else:
        data = np.load("reconstruction_debugging/training_samples/separated_training_samples.npz")
        all_latents = torch.tensor(data["latents"])
        all_images = torch.tensor(data["images"])

    model = ReconstructionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    print("Starting Training")
    for e in trange(epochs):
        epoch_losses = []
        idxs = np.arange(len(all_latents))
        # np.random.shuffle(idxs)
        # all_latents = all_latents[idxs]
        # all_images = all_images[idxs]
        for i_data in range(0, len(all_latents), batch_size):
            latents = all_latents[i_data:i_data+batch_size].to(device)
            images = all_images[i_data:i_data+batch_size].to(device)

            reconstructions = model(latents)
            loss = F.l1_loss(reconstructions, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print("EPOCH LOSS", np.mean(epoch_losses))
        concatted = np.transpose(np.concatenate((images[:30].detach().cpu().numpy(), reconstructions[:30].detach().cpu().numpy()), axis=3), (0, 2, 3, 1))
        for i_write in range(30):
            cv2.imwrite("reconstruction_debugging/visualizations/output" + str(i_write) + ".png", concatted[i_write])


if __name__ == "__main__":
    main()
