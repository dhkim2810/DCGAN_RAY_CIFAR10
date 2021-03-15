import ray

import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import entropy

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_data_loader():
    dataset = dset.MNIST(
        root=dataroot,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return dataloader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(dataloader):
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Original Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(real_batch[0][:64], padding=2,
                             normalize=True).cpu(), (1, 2, 0)))

    plt.show()


def inception_score(self, imgs, classifier, resize=False, batch_size=32, splits=1):
        """Calculate the inception score of the generated images."""
        N = len(imgs)
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
        up = nn.Upsample(
            size=(299, 299),
            mode="bilinear",
            align_corners=True
        ).type(torch.cuda.FloatTensor)

        def get_pred(x):
            if resize:
                x = up(x)
            x = classifier(x)
            return F.softmax(x, dim=1).data.cpu().numpy()

        # Obtain predictions for the fake provided images
        preds = np.zeros((N, 1000))
        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(torch.cuda.FloatTensor)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]
            preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

        # Now compute the mean kl-div
        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits):(k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)