<<<<<<< HEAD
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = '../Usual/dataset/CycleGAN/summer2winter/train'
TEST_DIR = '../Usual/dataset/CycleGAN/summer2winter/test'
SAMPLE_PATH = './samples'
BATCH_SIZE = 1
IMG_CHANNELS = 3
IMG_SIZE = 256
LEARNING_RATE = 2e-4
LAMBDA_CYCLE = 10
NUM_EPOCHS = 50
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PATH = './models'
RESULT_PATH = './results'
SAMPLE_INTERVAL = 600
CHECKPOINT_GEN_M = "GenSummer.pth"
CHECKPOINT_GEN_P = "GenWinter.pth"
CHECKPOINT_DISC_M = "DiscSummer.pth"
CHECKPOINT_DISC_P = "DiscWinter.pth"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

test_transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)
=======
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = '../Usual/dataset/CycleGAN/summer2winter/train'
TEST_DIR = '../Usual/dataset/CycleGAN/summer2winter/test'
SAMPLE_PATH = './samples'
BATCH_SIZE = 1
IMG_CHANNELS = 3
IMG_SIZE = 256
LEARNING_RATE = 2e-4
LAMBDA_CYCLE = 10
NUM_EPOCHS = 50
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PATH = './models'
RESULT_PATH = './results'
SAMPLE_INTERVAL = 600
CHECKPOINT_GEN_M = "GenSummer.pth"
CHECKPOINT_GEN_P = "GenWinter.pth"
CHECKPOINT_DISC_M = "DiscSummer.pth"
CHECKPOINT_DISC_P = "DiscWinter.pth"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

test_transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)
>>>>>>> 3cd70e3f9973e7cf3e699a7e9db30e87b361b0b9
