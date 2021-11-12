import torch


def add_noise(orig, noise_factor=0.5):
    noisy = orig + torch.randn_like(orig) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy
