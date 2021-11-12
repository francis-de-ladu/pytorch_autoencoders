from functools import partial

import torch
from torch import nn, optim
from utils.setup import get_device, load_dataset
from utils.train import train


class Encoder(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
        )

        self.flatten = nn.Flatten()

        self.encoder_lin = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(3 * 3 * 32, h_dim),
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(h_dim, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.name = 'autoencoder'
        self.encoder = Encoder(h_dim)
        self.decoder = Decoder(h_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DenoisingAE(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.name = 'denoising'
        self.encoder = Encoder(h_dim)
        self.decoder = Decoder(h_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ContractiveAE(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.name = 'contractive'
        self.encoder = Encoder(h_dim)
        self.decoder = Decoder(h_dim)

    def forward(self, x):
        self.hidden = self.encoder(x)
        x = self.decoder(self.hidden)
        return x


def contractive_loss(model, mse_loss, lamda):
    def loss_fn(orig, reconst):
        mse = mse_loss(orig, reconst)
        dh = model.hidden * (1 - model.hidden)
        W = model.state_dict()['encoder.encoder_lin.1.weight']
        W_squared_sum = W.pow(2).sum(dim=1).unsqueeze(1)
        contractive = torch.sum(torch.mm(dh.pow(2), W_squared_sum), dim=0)
        return mse + lamda * contractive
    return loss_fn


if __name__ == '__main__':
    device = get_device()
    print(f'Currently using `{device}` device.')

    torch.manual_seed(2021)
    train_loader, valid_loader, test_loader = load_dataset()

    model = AutoEncoder(16)
    model.to(device)

    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, device, loss_fn, optimizer,
          train_loader, valid_loader, epochs=20, noisy=False)
