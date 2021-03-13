import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
import numpy as np
from filelock import FileLock

from tqdm import trange

from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import entropy

import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.utils import override
from ray.util.sgd.torch import TrainingOperator

from model import Discriminator, Generator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GANOperator(TrainingOperator):
    def setup(self, config):
        discriminator = Discriminator(features=config.get("img_size", 64), num_channels=3).to("cuda:0")
        discriminator.apply(weights_init)

        generator = Generator(latent_vector_size=config.get("latent_vector_size", 100), features=config.get("img_size", 64), num_channels=3).to("cuda:0")
        generator.apply(weights_init)
        models = (discriminator, generator)

        discriminator_opt = optim.Adam(
            discriminator.parameters(),
            lr=config.get("lr", 2e-4),
            betas=(0.5, 0.999))
        generator_opt = optim.Adam(
            generator.parameters(),
            lr=config.get("lr", 0.01),
            betas=(0.5, 0.999))
        optimizers = (discriminator_opt, generator_opt)

        with FileLock(".ray.lock"):
            dataset = datasets.CIFAR10(root=config.get("data_dir","/dataset/CIFAR"),
                        transform=transforms.Compose([
                            transforms.Resize(config.get("img_size", 64)),
                            transforms.CenterCrop(config.get("img_size", 64)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ]), download=True)
        if config.get("test_mode"):
            dataset = torch.utils.data.Subset(dataset, list(range(config.get("test_bs", 25))))
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.get("batch_size", 128))

        self.models, self.optimizers, self.criterion = self.register(
            models=models, optimizers=optimizers, criterion=nn.BCELoss())
        self.register_data(
            train_loader=train_dataloader, validation_loader=None)

        self.model = self.models[0]
        self.optimizer = self.optimizers[0]

        self.classifier = inception_v3(pretrained=True, transform_input=False).type(torch.cuda.FloatTensor)
        self.classifier.eval()

        # self.ratio = config.get("update_ratio", 5) # update ratio for GAN training
        # self.count = 0

    def inception_score(self, imgs, resize=False, batch_size=32, splits=1):
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
            x = self.classifier(x)
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

    @override(TrainingOperator)
    def train_batch(self, batch, batch_info):
        """Trains on one batch of data from the data creator."""
        real_label = 1.0
        fake_label = 0.
        discriminator, generator = self.models
        optimD, optimG = self.optimizers

        # Compute a discriminator update for real images
        discriminator.zero_grad()
        # self.device is set automatically
        real_cpu = batch[0].to("cuda:0")
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size, ), real_label, device="cuda:0")
        output = discriminator(real_cpu).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()

        # Compute a discriminator update for fake images
        noise = torch.randn(
            batch_size,
            self.config.get("latent_vector_size", 100),
            1,
            1,
            device="cuda:0")
        fake = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake

        # Update the discriminator
        optimD.step()

        # Update the generator
        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        optimG.step()

        is_score, is_std = self.inception_score(fake, resize=True, splits=10)

        return {
            "loss_g": errG.item(),
            "loss_d": errD.item(),
            "inception": is_score,
            "num_samples": batch_size
        }


def train_example(num_workers=1, use_gpu=False, test_mode=False, args):
    config = {
        "data_dir" : args.data_dir,
        "img_size" : args.img_size,
        "test_mode": test_mode,
        "batch_size": args.test_Bs if test_mode else args.bs,
        "lr" : args.lr,
        "update_ratio" : args.update_ratio
    }
    trainer = TorchTrainer(
        training_operator_cls=GANOperator,
        num_workers=num_workers,
        config=config,
        use_gpu=use_gpu,
        use_tqdm=True)


    if not os.path.exists(os.path.join(os.getcwd(), "checkpoint", args.trial)):
        os.mkdir(os.path.join(os.getcwd(), "checkpoint", args.trial))
    LOSS = []
    from tabulate import tabulate
    pbar = trange(20, unit="epoch")
    for itr in pbar:
        stats = trainer.train(info=dict(epoch_idx=itr, num_epochs=20))
        LOSS.append(stats)
        pbar.set_postfix(dict(loss_g=stats["loss_g"], loss_d=stats["loss_d"], IS=stats["inception"]))
        formatted = tabulate([stats], headers="keys")
        if itr > 0:  # Get the last line of the stats.
            formatted = formatted.split("\n")[-1]
        pbar.write(formatted)

        torch.save(LOSS, os.path.join(os.getcwd(), "checkpoint", args.trial, "epoch_{}.loss".format(itr)))
        torch.save(trainer.get_model(), os.path.join(os.getcwd(), "checkpoint", args.trial, "model_{}.ray".format(itr)))

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use to connect to a cluster.")
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Sets number of workers for training.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--trial",
        type=int, default=1
    )
    parser.add_argument(
        "--data_dir",
        type=str, default="/dataset/CIFAR"
    )
    parser.add_argument(
        "img_size",
        type=int, default=64
    )
    parser.add_argument(
        "--bs",
        type=int, default=128
    )
    parser.add_argument(
        "--test_bs",
        type=int, default=16
    )
    parser.add_argument(
        "--lr",
        type=float, default=2e-4
    )
    parser.add_argument(
        "--update_ratio",
        type=int, default=1
    )
    args = parser.parse_args()
    if args.smoke_test:
        ray.init(num_cpus=2)
    else:
        ray.init(address=args.address)

    trainer = train_example(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        test_mode=args.smoke_test,
        args=args)
    models = trainer.get_model()