import torch
from autoencoder import AutoEncoder
from torch import nn, optim
from utils.setup import get_device, load_dataset
from utils.train import train

if __name__ == '__main__':
    device = get_device()
    print(f'Currently using `{device}` device.')

    torch.manual_seed(2021)
    train_loader, valid_loader, test_loader = load_dataset()

    model = AutoEncoder(256)
    model.to(device)

    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, device, loss_fn, optimizer,
          train_loader, valid_loader, epochs=20, noisy=True)
