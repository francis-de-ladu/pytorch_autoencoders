import argparse
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
    def __init__(self, h_dim, name):
        super().__init__()
        self.name = name
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


def sparse_loss(model, mse_loss, lamda):
    def loss_fn(orig, reconst):
        mse = mse_loss(orig, reconst)
        h_l1 = model.hidden.abs().sum(dim=1)
        sparse = torch.sum(h_l1, dim=0)
        return mse + lamda * sparse
    return loss_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train some encoder model on the MNIST dataset.')
    parser.add_argument('--h_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)

    subparsers = parser.add_subparsers(dest='ae')
    vanilla_parser = subparsers.add_parser('vanilla')
    denoising_parser = subparsers.add_parser('denoising')
    contractive_parser = subparsers.add_parser('contractive')
    sparse_parser = subparsers.add_parser('sparse')

    denoising_parser.add_argument('--noise', type=float, default=0.5)
    contractive_parser.add_argument(
        '--lambda', type=float, dest='lamda', default=1e-3)
    contractive_parser.add_argument('--noise', type=float, default=0)
    sparse_parser.add_argument(
        '--lambda', type=float, dest='lamda', default=1e-3)
    sparse_parser.add_argument('--noise', type=float, default=0)

    args = parser.parse_args()
    print(args)

    # get device (cuda if available)
    device = get_device()
    print(f'Currently using `{device}` device.')

    # set seed and get splits
    torch.manual_seed(2021)
    train_loader, valid_loader, test_loader = load_dataset()

    # instantiate the model
    model = AutoEncoder(args.h_dim, name=args.ae)
    model.to(device)

    loss_fn = nn.MSELoss(reduction='sum')
    if args.ae == 'contractive':
        loss_fn = partial(contractive_loss(model, loss_fn, args.lamda))
    elif args.ae == 'sparse':
        loss_fn = partial(sparse_loss(model, loss_fn, args.lamda))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, device, loss_fn, optimizer, train_loader,
          valid_loader, epochs=args.epochs, noise=getattr(args, 'noise', 0))
