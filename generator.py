<<<<<<< HEAD
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwarg):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwarg)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwarg),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x) + x


class Generator(nn.Module):
    def __init__(self, img_channels, features=64, num_residual=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels,
                      features,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList([
            ConvBlock(features,
                      features * 2,
                      down=True,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            ConvBlock(features * 2,
                      features * 4,
                      down=True,
                      kernel_size=3,
                      stride=2,
                      padding=1)
        ])

        self.residual_blocks = nn.Sequential(
            *[ResBlock(features * 4) for _ in range(num_residual)]
        )

        self.up_blocks = nn.ModuleList([
            ConvBlock(features * 4,
                      features * 2,
                      down=False,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      output_padding=1),
            ConvBlock(features * 2,
                      features,
                      down=False,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      output_padding=1),
        ])
        # cvt2RGB
        self.last = nn.Conv2d(features,
                              img_channels,
                              kernel_size=7,
                              stride=1,
                              padding=3,
                              padding_mode='reflect')

    def forward(self, x):
        x = self.initial(x)

        for layer in self.down_blocks:
            x = layer(x)

        x = self.residual_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)

        x = self.last(x)
        return torch.tanh(x)


def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((1, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    fake = gen(x)
    print(gen)
    print(fake.shape)


if __name__ == "__main__":
    test()
=======
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwarg):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwarg)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwarg),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x) + x


class Generator(nn.Module):
    def __init__(self, img_channels, features=64, num_residual=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels,
                      features,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList([
            ConvBlock(features,
                      features * 2,
                      down=True,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            ConvBlock(features * 2,
                      features * 4,
                      down=True,
                      kernel_size=3,
                      stride=2,
                      padding=1)
        ])

        self.residual_blocks = nn.Sequential(
            *[ResBlock(features * 4) for _ in range(num_residual)]
        )

        self.up_blocks = nn.ModuleList([
            ConvBlock(features * 4,
                      features * 2,
                      down=False,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      output_padding=1),
            ConvBlock(features * 2,
                      features,
                      down=False,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      output_padding=1),
        ])
        # cvt2RGB
        self.last = nn.Conv2d(features,
                              img_channels,
                              kernel_size=7,
                              stride=1,
                              padding=3,
                              padding_mode='reflect')

    def forward(self, x):
        x = self.initial(x)

        for layer in self.down_blocks:
            x = layer(x)

        x = self.residual_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)

        x = self.last(x)
        return torch.tanh(x)


def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((1, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    fake = gen(x)
    print(gen)
    print(fake.shape)


if __name__ == "__main__":
    test()
>>>>>>> 3cd70e3f9973e7cf3e699a7e9db30e87b361b0b9
