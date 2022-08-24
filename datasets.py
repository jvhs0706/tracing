import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

C, H, W = 3, 32, 32

# CIFAR10
CIFAR10_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

CIFAR10_transform_test = transforms.ToTensor()

CIFAR10_means, CIFAR10_stds = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

CIFAR10_training_set = torchvision.datasets.CIFAR10(root = f'/ssddata1/data', train = True, transform = CIFAR10_transform_train, download = False)
CIFAR10_testing_set = torchvision.datasets.CIFAR10(root = f'/ssddata1/data', train = False, transform = CIFAR10_transform_test, download = False)
CIFAR10_num_classes = 10

# GTSRB
GTSRB_transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

GTSRB_transform_test = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ToTensor()
])

GTSRB_means, GTSRB_stds = (0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)

GTSRB_training_set = torchvision.datasets.GTSRB(root = f'/ssddata1/data', split = 'train', transform = GTSRB_transform_train, download = False)
GTSRB_testing_set = torchvision.datasets.GTSRB(root = f'/ssddata1/data', split = 'test', transform = GTSRB_transform_test, download = False)
GTSRB_num_classes = 43