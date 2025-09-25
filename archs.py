# filename: archs.py

from dataset import get_normalize_layer
import torch

def get_archs(arch, dataset='cifar10'):
    """
    Loads a pre-trained model architecture and wraps it with a normalization layer.
    """
    # Note: The original normalization is now part of the dataset transform for evaluation.
    # The get_normalize_layer is kept in case direct, un-normalized inputs are ever used.
    # However, for simplicity, we will instantiate the model without the extra layer,
    # as the test dataset now provides correctly normalized images.
    
    if dataset == 'cifar10':
        if arch == 'resnet56_cifar10':
            # Load the pre-trained CIFAR-10 ResNet-56 model
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
        else:
            raise ValueError(f"Unsupported architecture '{arch}' for CIFAR-10. Use 'resnet56_cifar10'.")
    else:
        raise ValueError(f"Unsupported dataset '{dataset}' in get_archs.")

    # The dataset loader for the 'test' split now handles normalization.
    # Returning the raw model is sufficient.
    return model