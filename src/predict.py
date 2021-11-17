import numpy as np
import torch
from autoencoders import AutoEncoder
from torch import autograd, nn, optim
from tqdm import tqdm
from utils.data import load_dataset
from utils.misc import get_device


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(16, 10)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return self.sm(x)


if __name__ == '__main__':
    device = get_device()

    model = AutoEncoder(16, name='denoising')
    model.load_state_dict(torch.load(f'../models/{model.name}.model'))

    for params in model.parameters():
        params.requires_grad = False

    model = Classifier(model.encoder)
    model.to(device)

    # set seed and get splits
    torch.manual_seed(2021)
    train_loader, valid_loader, test_loader = load_dataset()

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.AdamW(model.parameters(), lr=5e-3)

    epochs = 20
    for epoch in tqdm(range(epochs)):
        model.train()
        batch_losses = []
        for data, labels in train_loader:
            # send inputs to device
            data = data.to(device)
            labels = labels.to(device)

            # forward pass
            probs = model(data)
            loss = loss_fn(probs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct_preds = 0
        for data, labels in valid_loader:
            # send inputs to device
            data = data.to(device)
            labels = labels.to(device)

            # forward pass
            probs = model(data)

            # compute accuracy
            preds = torch.argmax(probs, dim=1)
            correct_preds += torch.sum(preds == labels)

        accuracy = correct_preds / len(valid_loader.dataset)
        print(f'Validation accuracy: {accuracy}')
