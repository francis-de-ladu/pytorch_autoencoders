import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def load_dataset(data_path='../data', batch_size=64, valid_size=5000, normalize=False):
    if normalize is True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.ToTensor()

    train = torchvision.datasets.MNIST(
        data_path, train=True, download=True, transform=transform)
    train, valid = train_test_split(
        train, test_size=valid_size, random_state=42)

    test = torchvision.datasets.MNIST(
        data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader