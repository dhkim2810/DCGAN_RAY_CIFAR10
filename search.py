import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

import argparse
import os
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
import numpy as np

from model import Discriminator, Generator #, demo_gan
from utils import weights_init, inception_score


TRAIN_ITERATIONS_PER_STEP = 5


def train(netD, netG, optimG, optimD, criterion, dataloader, iteration, device, model_ref):
    real_label = 1.
    fake_label = 0.

    for i, data in enumerate(dataloader, 0):
        if i >= TRAIN_ITERATIONS_PER_STEP:
            break

        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size, ), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimG.step()

        is_score, is_std = inception_score(imgs=fake, model_ref=model_ref, resize=True, splits=10)

        # Output training stats
        if iteration % 10 == 0:
            print("[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z))"
                  ": %.4f / %.4f \tInception score: %.4f" %
                  (iteration, len(dataloader), errD.item(), errG.item(), D_x,
                   D_G_z1, D_G_z2, is_score))

    return errG.item(), errD.item(), is_score

# __Train_begin__
def dcgan_train(config):
    step = 0
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    discriminator = Discriminator(
        features=config.get("img_size", 64),
        num_channels=3).to(device)
    discriminator.apply(weights_init)

    generator = Generator(
        latent_vector_size=config.get("latent_vector_size", 100),
        features=config.get("img_size", 64),
        num_channels=3).to(device)
    generator.apply(weights_init)

    criterion = nn.BCELoss()
    discriminator_opt = optim.Adam(
        discriminator.parameters(), lr=config.get("lr", 0.01), betas=(0.5, 0.999))
    generator_opt = optim.Adam(
        generator.parameters(), lr=config.get("lr", 0.01), betas=(0.5, 0.999))

    with FileLock(".ray.lock"):
        dataset = datasets.CIFAR10(root=config.get("data_dir","/dataset/CIFAR"),
                    transform=transforms.Compose([
                        transforms.Resize(config.get("img_size", 64)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]), download=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.get("batch_size", 128), shuffle=True, num_workers=2)

    if config["resume"]:
        path = os.path.join(config["checkpoint_dir"], str(config["trial"]))
        checkpoint = torch.load(path)
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator.load_state_dict(checkpoint["generator"])
        discriminator_opt.load_state_dict(checkpoint["discriminator_opt"])
        generator_opt.load_state_dict(checkpoint["generator_opt"])
        step = checkpoint["step"]

        if "lr_D" in config:
            for param_group in discriminator_opt.param_groups:
                param_group["lr"] = config["lr_D"]
        if "lr_G" in config:
            for param_group in generator_opt.param_groups:
                param_group["lr"] = config["lr_G"]

    while True:
        lossG, lossD, is_score = train(discriminator, generator, generator_opt, discriminator_opt,
                                       criterion, dataloader, step, device,
                                       config["model_ref"])
        step += 1
        with tune.checkpoint_dir(step=step) as config["checkpoint_dir"]:
            path = os.path.join(config["checkpoint_dir"], str(config["trial"]))
            if not os.path.exists(path):
                os.mkdir(config["checkpoint_dir"])
            torch.save({
                "discriminator": discriminator.state_dict(),
                "generator": generator.state_dict(),
                "discriminator_opt": discriminator_opt.state_dict(),
                "generator_opt": generator_opt.state_dict(),
                "step": step,
            }, path)
        tune.report(lossg=lossG, lossd=lossD, is_score=is_score)


# __Train_end__

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
        help="Finish quickly for testing")
    parser.add_argument("--address", required=False, type=str,
        help="the address to use to connect to a cluster.")
    parser.add_argument("--num-workers", "-n", type=int, default=1,
        help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False,
        help="Enables GPU training")
    parser.add_argument("--resume", action="store_true", default=False,
        help="Resume training from last checkpoint")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="/dataset/CIFAR")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/search")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--test_bs", type=int, default=16)
    args = parser.parse_args()
    ray.init()

    """dataloader = get_data_loader()
    if not args.smoke_test:
        plot_images(dataloader)"""

    # __tune_begin__

    # load the pretrained mnist classification model for inception_score
    classifier = inception_v3(pretrained=True, transform_input=False).type(torch.cuda.FloatTensor)
    classifier.eval()
    # Put the model in Ray object store.
    model_ref = ray.put(classifier)

    scheduler = PopulationBasedTraining(
        perturbation_interval=5,
        hyperparam_mutations={
            # distribution for resampling
            "lr_G": lambda: np.random.uniform(1e-2, 1e-5),
            "lr_D": lambda: np.random.uniform(1e-2, 1e-5),
        })

    tune_iter = 5 if args.smoke_test else 300
    analysis = tune.run(
        dcgan_train,
        name="pbt_dcgan",
        scheduler=scheduler,
        verbose=1,
        stop={
            "training_iteration": tune_iter,
        },
        resources_per_trial={'gpu':1,'cpu':2},
        metric="is_score",
        mode="max",
        num_samples=8,
        config={
            "resume" : args.resume,
            "trial" : args.trial,
            "data_dir" : args.data_dir,
            "checkpoint_dir" : args.checkpoint_dir,
            "img_size" : args.img_size,
            "lr_G": tune.choice([0.0001, 0.0002, 0.0005]),
            "lr_D": tune.choice([0.0001, 0.0002, 0.0005]),
            "model_ref": model_ref
        })
    # __tune_end__
"""
    # demo of the trained Generators
    if not args.smoke_test:
        all_trials = analysis.trials
        checkpoint_paths = [
            os.path.join(analysis.get_best_checkpoint(t), "checkpoint")
            for t in all_trials
        ]
        demo_gan(analysis, args)
"""