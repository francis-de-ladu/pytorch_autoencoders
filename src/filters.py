import os
from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from autoencoder import AutoEncoder
from torch import Tensor, autograd, nn, optim
from utils.setup import get_device


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
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

    model = AutoEncoder(256)
    model.load_state_dict(torch.load('../models/denoising.model'))
    model.to(device)
    model.eval()

    for params in model.parameters():
        params.requires_grad = False

    conv_layers = extract_conv_layers(model)
    feat_extractor = FeatureExtractor(model, conv_layers)

    base_img = np.random.uniform(.2, .8, (1, 28, 28))

    lr = 0.1
    upsampling = 1.25

    save_dir = '../filters'
    os.makedirs(save_dir, exist_ok=True)

    for layer_pos, layer in enumerate(conv_layers):
        # scope = [3, 5, 12, 14, 28][layer_pos]
        # num_upsampling = int(np.round(np.log(28 / scope) / np.log(upsampling)))
        # print(num_upsampling)
        num_upsampling = 5
        for filtr in range(8):
            img = Tensor(base_img)
            for i in range(num_upsampling + 1):
                for step in range(100):
                    # preparation
                    img_var = autograd.Variable(img[None], requires_grad=True)
                    optimizer = optim.Adam(
                        [img_var], lr=lr, weight_decay=1e-6)

                    # forward pass
                    feat_extractor.model(img_var)
                    loss = -feat_extractor._features[layer][0, filtr].mean()

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                img = img_var.detach().cpu().numpy()[0]
                new_size = int(img.shape[-1] * upsampling)
                delta = (new_size - 28) // 2
                img[0] = cv2.resize(
                    img[0],
                    (new_size, new_size),
                    interpolation=cv2.INTER_CUBIC
                )[delta:-(delta + 1), delta:-(delta + 1)]
                img[0] = cv2.blur(img[0], (4, 4))
                img = Tensor(img)

            # plot filter state
            img_np = img_var.detach().cpu().numpy()[0][0]
            save_path = f'{save_dir}/layer_{layer}__filter_{filtr}.png'
            plt.imshow(img_np, cmap='gray')
            plt.savefig(save_path)
