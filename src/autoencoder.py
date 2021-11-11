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
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten()

        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 128, 2048),
            nn.ReLU(True),
            nn.Linear(2048, h_dim)
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
            nn.Linear(h_dim, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 3 * 3 * 128),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                16, 8, 3, stride=1, padding=1, output_padding=0),
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
        self.encoder = Encoder(h_dim)
        self.decoder = Decoder(h_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    device = get_device()
    print(f'Currently using `{device}` device.')

    torch.manual_seed(2021)
    train_loader, valid_loader, test_loader = load_dataset()

    model = AutoEncoder(8)
    model.to(device)

    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, device, loss_fn, optimizer,
          train_loader, valid_loader, epochs=20, noisy=False)
