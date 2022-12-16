<<<<<<< HEAD
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Initial
            # img_channels -> 64
            nn.Conv2d(img_channels,
                      features[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        in_channels = features[0]
        for feature in features[1:]:
            self.disc.append(self.block(in_channels,
                                        feature,
                                        stride=1 if feature == features[-1] else 2))
            in_channels = feature

        self.disc.append(
            nn.Conv2d(in_channels,
                      1,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      padding_mode='reflect')
        )
        self.disc.append(nn.Sigmoid())

    def block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=4,
                      stride=stride,
                      padding=1,
                      bias=True,
                      padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(3)
    pred = model(x)
    print(model)
    print(pred.shape)


if __name__ == "__main__":
    test()
=======
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Initial
            # img_channels -> 64
            nn.Conv2d(img_channels,
                      features[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        in_channels = features[0]
        for feature in features[1:]:
            self.disc.append(self.block(in_channels,
                                        feature,
                                        stride=1 if feature == features[-1] else 2))
            in_channels = feature

        self.disc.append(
            nn.Conv2d(in_channels,
                      1,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      padding_mode='reflect')
        )
        self.disc.append(nn.Sigmoid())

    def block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=4,
                      stride=stride,
                      padding=1,
                      bias=True,
                      padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(3)
    pred = model(x)
    print(model)
    print(pred.shape)


if __name__ == "__main__":
    test()
>>>>>>> 3cd70e3f9973e7cf3e699a7e9db30e87b361b0b9
