# filename: dataset.py

from torch.utils.data import Dataset
from torchvision import transforms, datasets
from typing import List
import torch

# --- Global Constants ---
# Normalization for DDPM models (maps images to [-1, 1])
_CIFAR10_MEAN = [0.5, 0.5, 0.5]
_CIFAR10_STDDEV = [0.5, 0.5, 0.5]

# Normalization for the pre-trained ResNet-56 classifier
_CLASSIFIER_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CLASSIFIER_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


# --- Dataset Loading ---

def get_dataset(dataset_name: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object."""
    if dataset_name == 'cifar10':
        return _cifar10(split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def _cifar10(split: str) -> Dataset:
    """
    Loads the CIFAR-10 dataset.
    - For training the purifier ('train' split), normalizes to [-1, 1].
    - For evaluating the classifier ('test' split), uses the classifier's specific normalization.
    """
    is_train = (split == "train")
    if is_train:
        # Normalization for the U-Net purifier model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
        ])
    else:
        # Normalization for the pre-trained victim classifier
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CLASSIFIER_CIFAR10_MEAN, _CLASSIFIER_CIFAR10_STDDEV)
        ])
    
    return datasets.CIFAR10(root="./datasets", train=is_train, download=True, transform=transform)


# --- Metadata ---

def get_num_classes(dataset: str) -> int:
    """Return the number of classes in the dataset."""
    if 'cifar10' in dataset:
        return 10
    return 1000

# --- Normalization Layer ---

def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer for the classifier."""
    if "cifar10" in dataset:
        return NormalizeLayer(_CLASSIFIER_CIFAR10_MEAN, _CLASSIFIER_CIFAR10_STDDEV)
    else:
        # Default fallback for other potential datasets, adjust as needed.
        return NormalizeLayer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class NormalizeLayer(torch.nn.Module):
    """Standard normalization layer."""
    def __init__(self, means: List[float], sds: List[float]):
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means).view(1, -1, 1, 1))
        self.register_buffer('sds', torch.tensor(sds).view(1, -1, 1, 1))

    def forward(self, input: torch.tensor):
        return (input - self.means) / self.sds