# The FashionMNIST features are in PIL Image format, and the labels are
# integers. For training, we need the features as normalized tensors, and the
# labels as one-hot encoded tensors. To make these transformations, we use
# ToTensor and Lambda.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, \
        dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
