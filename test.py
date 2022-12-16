<<<<<<< HEAD
import os

import torchvision.utils

from generator import Generator
import torchvision.transforms as transforms
import torch
import config
from dataset import testDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import *


def test(gen, loader):
    with torch.no_grad():
        for idx, img in enumerate(loader):
            output = gen(img)
            output = output * 0.5 + 0.5
            torchvision.utils.save_image(output, "{:s}/{:d}.png".format(config.RESULT_PATH, idx + 1))
            print("%d" % (idx + 1))


def main():
    gen = Generator(config.IMG_CHANNELS)
    gen1 = Generator(config.IMG_CHANNELS)
    opt_gen = optim.Adam(list(gen.parameters()) + list(gen1.parameters()),
                         lr=config.LEARNING_RATE,
                         betas=(0.5, 0.999))
    load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_P, gen, opt_gen, config.LEARNING_RATE)
    dataset = testDataset(config.TEST_DIR, transforms=config.test_transforms)
    loader = DataLoader(dataset=dataset, shuffle=False, batch_size=config.BATCH_SIZE)
    test(gen, loader)


if __name__ == "__main__":
    main()
=======
import os

import torchvision.utils

from generator import Generator
import torchvision.transforms as transforms
import torch
import config
from dataset import testDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import *


def test(gen, loader):
    with torch.no_grad():
        for idx, img in enumerate(loader):
            output = gen(img)
            output = output * 0.5 + 0.5
            torchvision.utils.save_image(output, "{:s}/{:d}.png".format(config.RESULT_PATH, idx + 1))
            print("%d" % (idx + 1))


def main():
    gen = Generator(config.IMG_CHANNELS)
    gen1 = Generator(config.IMG_CHANNELS)
    opt_gen = optim.Adam(list(gen.parameters()) + list(gen1.parameters()),
                         lr=config.LEARNING_RATE,
                         betas=(0.5, 0.999))
    load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_P, gen, opt_gen, config.LEARNING_RATE)
    dataset = testDataset(config.TEST_DIR, transforms=config.test_transforms)
    loader = DataLoader(dataset=dataset, shuffle=False, batch_size=config.BATCH_SIZE)
    test(gen, loader)


if __name__ == "__main__":
    main()
>>>>>>> 3cd70e3f9973e7cf3e699a7e9db30e87b361b0b9
