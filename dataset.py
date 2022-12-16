<<<<<<< HEAD
import torch
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
import config
import numpy as np


class SummerWinterDataset(Dataset):
    def __init__(self, root_Summer, root_Winter, transforms=None):
        super(SummerWinterDataset, self).__init__()
        self.root_Summer = root_Summer
        self.root_Winter = root_Winter
        self.transform = transforms

        self.SummerImages = os.listdir(self.root_Summer)
        self.WinterImages = os.listdir(self.root_Winter)
        self.len = max(len(self.SummerImages), len(self.WinterImages))
        self.Summer_len = len(self.SummerImages)
        self.Winter_len = len(self.WinterImages)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        summer = self.SummerImages[index % self.Summer_len]
        winter = self.WinterImages[index % self.Winter_len]
        summer_path = self.root_Summer + '/' + summer
        winter_path = self.root_Winter + '/' + winter
        summer, winter = Image.open(summer_path).convert('RGB'), Image.open(winter_path).convert('RGB')
        summer, winter = np.asarray(summer), np.asarray(winter)

        if self.transform:
            augs = self.transform(image=summer, image0=winter)
            summer, winter = augs['image'], augs['image0']

        return summer, winter


class testDataset(Dataset):
    def __init__(self, root, transforms=None):
        super(testDataset, self).__init__()
        self.root = root
        self.transforms = transforms

        self.images = os.listdir(self.root)
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = self.images[index]
        img = self.root + '/' + img
        img = Image.open(img).convert("RGB")
        img = np.asarray(img)
        if self.transforms:
            augs = self.transforms(image=img)
            img = augs['image']

        return img
=======
import torch
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
import config
import numpy as np


class SummerWinterDataset(Dataset):
    def __init__(self, root_Summer, root_Winter, transforms=None):
        super(SummerWinterDataset, self).__init__()
        self.root_Summer = root_Summer
        self.root_Winter = root_Winter
        self.transform = transforms

        self.SummerImages = os.listdir(self.root_Summer)
        self.WinterImages = os.listdir(self.root_Winter)
        self.len = max(len(self.SummerImages), len(self.WinterImages))
        self.Summer_len = len(self.SummerImages)
        self.Winter_len = len(self.WinterImages)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        summer = self.SummerImages[index % self.Summer_len]
        winter = self.WinterImages[index % self.Winter_len]
        summer_path = self.root_Summer + '/' + summer
        winter_path = self.root_Winter + '/' + winter
        summer, winter = Image.open(summer_path).convert('RGB'), Image.open(winter_path).convert('RGB')
        summer, winter = np.asarray(summer), np.asarray(winter)

        if self.transform:
            augs = self.transform(image=summer, image0=winter)
            summer, winter = augs['image'], augs['image0']

        return summer, winter


class testDataset(Dataset):
    def __init__(self, root, transforms=None):
        super(testDataset, self).__init__()
        self.root = root
        self.transforms = transforms

        self.images = os.listdir(self.root)
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = self.images[index]
        img = self.root + '/' + img
        img = Image.open(img).convert("RGB")
        img = np.asarray(img)
        if self.transforms:
            augs = self.transforms(image=img)
            img = augs['image']

        return img
>>>>>>> 3cd70e3f9973e7cf3e699a7e9db30e87b361b0b9
