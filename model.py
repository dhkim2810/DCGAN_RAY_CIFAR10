import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Generator(nn.Module):
    def build_layer (self, latent_vector_size, features, num_channels):
        layers = []
        if features == 64:
            layers.extend([
                nn.ConvTranspose2d(latent_vector_size, features * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features * 8), nn.ReLU(True),
                nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 4), nn.ReLU(True)
            ])
        else :
            layers.extend([
                nn.ConvTranspose2d(latent_vector_size, features * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features * 4), nn.ReLU(True),
            ])
        layers.extend([
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2), nn.ReLU(True),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features), nn.ReLU(True),
            nn.ConvTranspose2d(features, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        ])
        return nn.Sequential(*layers)

    def __init__(self, latent_vector_size, features=32, num_channels=1):
        super(Generator, self).__init__()
        self.main = self.build_layer(latent_vector_size, features, num_channels)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def build_layer (self, features, num_channels):
        layers = [
            nn.Conv2d(num_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4), nn.LeakyReLU(0.2, inplace=True)
            ]
        if features == 64:
            layers.extend([
                nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 8), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
            ])
        else :
            layers.extend([
                nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False), nn.Sigmoid()
            ])
        return nn.Sequential(*layers)

    def __init__(self, features=32, num_channels=1):
        super(Discriminator, self).__init__()
        self.main = self.build_layer(features, num_channels)

    def forward(self, x):
        return self.main(x)


"""def demo_gan(args, epoch):
    img_list = []
    fixed_noise = torch.randn(args.bs, 100, 1, 1)
    
    loadedG = Generator()
    loadedG.load_state_dict(torch.load(os.path.join(args.checkpoint, "netG_{}.ray".format(epoch)))
    with torch.no_grad():
        fake = loadedG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save("./generated.gif", writer="imagemagick", dpi=72)"""