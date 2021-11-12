import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from autoencoders import AutoEncoder
from torch import autograd, nn, optim
from utils.setup import get_device


class FeatureExtractor(nn.Module):
    def __init__(self, model, device, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features


def extract_conv_layers(model):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
    return conv_layers


if __name__ == '__main__':
    device = get_device()

    model = AutoEncoder(16, name='sparse')
    model.load_state_dict(torch.load(f'../models/{model.name}.model'))
    model.to(device)
    model.eval()

    for params in model.parameters():
        params.requires_grad = False

    conv_layers = extract_conv_layers(model)
    feat_extractor = FeatureExtractor(model, device, conv_layers)
    feat_extractor.to(device)

    base_img = np.random.uniform(.2, .8, (1, 28, 28))

    lr = 0.01
    upsampling = 1.2

    save_dir = '../filters'
    os.makedirs(save_dir, exist_ok=True)

    for layer_pos, layer in enumerate(conv_layers):
        num_upsampling = 6
        num_filters = feat_extractor._features[layer].shape[1] if len(
            feat_extractor._features[layer].shape) > 1 else 8
        for filtr in range(num_filters):
            upsamplings = np.ndarray([28, (num_upsampling + 1) * 28])
            upsamplings[:, :28] = base_img
            img = torch.Tensor(base_img).to(device)
            for i in range(num_upsampling):
                for step in range(25):
                    # preparation
                    img_var = autograd.Variable(img[None], requires_grad=True)
                    img_var.to(device)
                    optimizer = optim.Adam([img_var], lr=lr, weight_decay=1e-6)

                    # forward pass
                    feat_extractor.model(img_var)
                    loss = -feat_extractor._features[layer][0, filtr].mean()

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                img = img_var.detach().cpu().numpy()[0]
                upsamplings[:, (i + 1) * 28:(i + 2) * 28] = img[0]
                new_size = int(img.shape[-1] * upsampling)
                delta = (new_size - 28) // 2
                img[0] = cv2.resize(
                    img[0],
                    (new_size, new_size),
                    interpolation=cv2.INTER_CUBIC
                )[delta:-(delta + 1), delta:-(delta + 1)]
                img[0] = cv2.blur(img[0], (5, 5))
                img = torch.Tensor(img).to(device)

            # plot filter state
            img_np = img_var.detach().cpu().numpy()[0][0]
            save_path = f'{save_dir}/layer_{layer}__filter_{filtr}.png'
            plt.imshow(upsamplings, cmap='gray', vmin=0, vmax=1)
            plt.title(f'Layer_{layer_pos} : Filter_{filtr}')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_path)
