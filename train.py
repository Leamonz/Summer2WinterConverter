<<<<<<< HEAD
import torch
import torchvision.utils

from dataset import SummerWinterDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from discriminator import Discriminator
from generator import Generator
from time import time
from tqdm import tqdm


def train(disc_M, disc_P, gen_M, gen_P, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler, epoch):
    count: int = 1
    loop = tqdm(loader, leave=True)
    for batch_idx, (summer, winter) in enumerate(loop):
        summer, winter = summer.to(config.DEVICE), winter.to(config.DEVICE)

        # 训练Discriminator
        with torch.cuda.amp.autocast():
            fake_winter = gen_P(summer)
            D_Winter_Real = disc_P(winter)
            D_Winter_Fake = disc_P(fake_winter.detach())
            loss_D_Winter_Real = MSE(D_Winter_Real, torch.ones_like(D_Winter_Real))
            loss_D_Winter_Fake = MSE(D_Winter_Fake, torch.zeros_like(D_Winter_Fake))
            loss_D_Winter = loss_D_Winter_Fake + loss_D_Winter_Real

            fake_summer = gen_M(winter)
            D_Summer_Real = disc_M(summer)
            D_Summer_Fake = disc_M(fake_summer.detach())
            loss_D_Summer_Real = MSE(D_Summer_Real, torch.ones_like(D_Summer_Real))
            loss_D_Summer_Fake = MSE(D_Summer_Fake, torch.ones_like(D_Summer_Fake))
            loss_D_Summer = loss_D_Summer_Fake + loss_D_Summer_Real

            loss_D = (loss_D_Summer + loss_D_Winter) / 2

        opt_disc.zero_grad()
        d_scaler.scale(loss_D).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # 训练Generator
        with torch.cuda.amp.autocast():
            D_P_Fake = disc_P(fake_winter)
            D_M_Fake = disc_M(fake_summer)
            loss_G_Winter = MSE(D_P_Fake, torch.ones_like(D_P_Fake))
            loss_G_Summer = MSE(D_M_Fake, torch.ones_like(D_M_Fake))

            # cycle consistency loss & identity loss
            cycle_Summer = gen_M(fake_winter)
            cycle_Winter = gen_P(fake_summer)
            cycle_Summer_loss = L1(summer, cycle_Summer)
            cycle_Winter_loss = L1(winter, cycle_Winter)

            loss_G = loss_G_Summer \
                     + config.LAMBDA_CYCLE * cycle_Summer_loss \
                     + loss_G_Winter \
                     + config.LAMBDA_CYCLE * cycle_Winter_loss

        opt_gen.zero_grad()
        # 先scale，再用scale后的损失计算梯度
        g_scaler.scale(loss_G).backward()
        g_scaler.step(opt_gen)
        # 更新scaler factor
        g_scaler.update()

        if batch_idx % config.SAMPLE_INTERVAL == 0:
            print(f"[iterations:{batch_idx}/{len(loop)}], [loss_D:{loss_D.item()}], [loss_G:{loss_G.item()}]")
            # Sample Images
            torchvision.utils.save_image(fake_summer * 0.5 + 0.5,
                                         r'{:s}/epoch{:d}_summer_sample_{:d}.png'.format(config.SAMPLE_PATH, epoch + 1,
                                                                                         count))
            torchvision.utils.save_image(fake_winter * 0.5 + 0.5,
                                         r'{:s}/epoch{:d}_winter_sample_{:d}.png'.format(config.SAMPLE_PATH, epoch + 1,
                                                                                         count))
            count += 1


def main():
    disc_M = Discriminator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    disc_P = Discriminator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    gen_M = Generator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    gen_P = Generator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    opt_disc = optim.Adam(list(disc_P.parameters()) + list(disc_M.parameters()),
                          lr=config.LEARNING_RATE,
                          betas=(0.5, 0.999))
    opt_gen = optim.Adam(list(gen_P.parameters()) + list(gen_M.parameters()),
                         lr=config.LEARNING_RATE,
                         betas=(0.5, 0.999))
    # cycle consistency loss & identity loss
    L1 = nn.L1Loss()
    # generative adversarial loss
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_M, gen_M, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_P, gen_P, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_M, disc_M, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_P, disc_P, opt_disc, config.LEARNING_RATE)
    dataset = SummerWinterDataset(root_Summer=config.TRAIN_DIR + '/summer',
                                root_Winter=config.TRAIN_DIR + '/winter',
                                transforms=config.transforms)
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    # 自动混合精度
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.NUM_EPOCHS):
        print(f"=========Epoch{epoch + 1} Starts!=========")
        start = time()
        train(disc_M, disc_P, gen_M, gen_P, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler, epoch)

        if config.SAVE_MODEL:
            save_checkpoint(disc_M, opt_disc, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_M)
            save_checkpoint(disc_P, opt_disc, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_P)
            save_checkpoint(gen_M, opt_gen, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_M)
            save_checkpoint(gen_P, opt_gen, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_P)
        print(f"=========Epoch{epoch + 1} Ends!=========")
        end = time()
        print("Time taken: {:.2f}s".format(end - start))


if __name__ == "__main__":
    main()
=======
import torch
import torchvision.utils

from dataset import SummerWinterDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from discriminator import Discriminator
from generator import Generator
from time import time
from tqdm import tqdm


def train(disc_M, disc_P, gen_M, gen_P, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler, epoch):
    count: int = 1
    loop = tqdm(loader, leave=True)
    for batch_idx, (summer, winter) in enumerate(loop):
        summer, winter = summer.to(config.DEVICE), winter.to(config.DEVICE)

        # 训练Discriminator
        with torch.cuda.amp.autocast():
            fake_winter = gen_P(summer)
            D_Winter_Real = disc_P(winter)
            D_Winter_Fake = disc_P(fake_winter.detach())
            loss_D_Winter_Real = MSE(D_Winter_Real, torch.ones_like(D_Winter_Real))
            loss_D_Winter_Fake = MSE(D_Winter_Fake, torch.zeros_like(D_Winter_Fake))
            loss_D_Winter = loss_D_Winter_Fake + loss_D_Winter_Real

            fake_summer = gen_M(winter)
            D_Summer_Real = disc_M(summer)
            D_Summer_Fake = disc_M(fake_summer.detach())
            loss_D_Summer_Real = MSE(D_Summer_Real, torch.ones_like(D_Summer_Real))
            loss_D_Summer_Fake = MSE(D_Summer_Fake, torch.ones_like(D_Summer_Fake))
            loss_D_Summer = loss_D_Summer_Fake + loss_D_Summer_Real

            loss_D = (loss_D_Summer + loss_D_Winter) / 2

        opt_disc.zero_grad()
        d_scaler.scale(loss_D).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # 训练Generator
        with torch.cuda.amp.autocast():
            D_P_Fake = disc_P(fake_winter)
            D_M_Fake = disc_M(fake_summer)
            loss_G_Winter = MSE(D_P_Fake, torch.ones_like(D_P_Fake))
            loss_G_Summer = MSE(D_M_Fake, torch.ones_like(D_M_Fake))

            # cycle consistency loss & identity loss
            cycle_Summer = gen_M(fake_winter)
            cycle_Winter = gen_P(fake_summer)
            cycle_Summer_loss = L1(summer, cycle_Summer)
            cycle_Winter_loss = L1(winter, cycle_Winter)

            loss_G = loss_G_Summer \
                     + config.LAMBDA_CYCLE * cycle_Summer_loss \
                     + loss_G_Winter \
                     + config.LAMBDA_CYCLE * cycle_Winter_loss

        opt_gen.zero_grad()
        # 先scale，再用scale后的损失计算梯度
        g_scaler.scale(loss_G).backward()
        g_scaler.step(opt_gen)
        # 更新scaler factor
        g_scaler.update()

        if batch_idx % config.SAMPLE_INTERVAL == 0:
            print(f"[iterations:{batch_idx}/{len(loop)}], [loss_D:{loss_D.item()}], [loss_G:{loss_G.item()}]")
            # Sample Images
            torchvision.utils.save_image(fake_summer * 0.5 + 0.5,
                                         r'{:s}/epoch{:d}_summer_sample_{:d}.png'.format(config.SAMPLE_PATH, epoch + 1,
                                                                                         count))
            torchvision.utils.save_image(fake_winter * 0.5 + 0.5,
                                         r'{:s}/epoch{:d}_winter_sample_{:d}.png'.format(config.SAMPLE_PATH, epoch + 1,
                                                                                         count))
            count += 1


def main():
    disc_M = Discriminator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    disc_P = Discriminator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    gen_M = Generator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    gen_P = Generator(img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    opt_disc = optim.Adam(list(disc_P.parameters()) + list(disc_M.parameters()),
                          lr=config.LEARNING_RATE,
                          betas=(0.5, 0.999))
    opt_gen = optim.Adam(list(gen_P.parameters()) + list(gen_M.parameters()),
                         lr=config.LEARNING_RATE,
                         betas=(0.5, 0.999))
    # cycle consistency loss & identity loss
    L1 = nn.L1Loss()
    # generative adversarial loss
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_M, gen_M, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_P, gen_P, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_M, disc_M, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_P, disc_P, opt_disc, config.LEARNING_RATE)
    dataset = SummerWinterDataset(root_Summer=config.TRAIN_DIR + '/summer',
                                root_Winter=config.TRAIN_DIR + '/winter',
                                transforms=config.transforms)
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    # 自动混合精度
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.NUM_EPOCHS):
        print(f"=========Epoch{epoch + 1} Starts!=========")
        start = time()
        train(disc_M, disc_P, gen_M, gen_P, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler, epoch)

        if config.SAVE_MODEL:
            save_checkpoint(disc_M, opt_disc, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_M)
            save_checkpoint(disc_P, opt_disc, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_DISC_P)
            save_checkpoint(gen_M, opt_gen, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_M)
            save_checkpoint(gen_P, opt_gen, filename=config.MODEL_PATH + '/' + config.CHECKPOINT_GEN_P)
        print(f"=========Epoch{epoch + 1} Ends!=========")
        end = time()
        print("Time taken: {:.2f}s".format(end - start))


if __name__ == "__main__":
    main()
>>>>>>> 3cd70e3f9973e7cf3e699a7e9db30e87b361b0b9
