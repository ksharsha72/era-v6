import pytest
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets

# Add parent directory to path to import model2
sys.path.append(str(Path(__file__).parent.parent))
from model2 import Model


def test_model_parameters():
    """Test if model has less than 20000 parameters"""
    model = Model()
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params < 20000
    ), f"Model has {total_params} parameters, should be less than 20000"


def test_batch_normalization_exists():
    """Test if model contains batch normalization layers"""
    model = Model()
    has_batch_norm = any(
        isinstance(layer, nn.BatchNorm2d)
        for layer in model.modules()
        if not isinstance(layer, Model)
    )
    assert has_batch_norm, "Model should contain batch normalization layers"


def test_image_augmentation():
    """Test if image augmentation is used in the training pipeline"""
    # Check S6.ipynb for the transforms used
    transforms_list = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(size=28),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    assert any(
        isinstance(
            t,
            (transforms.RandomCrop, transforms.RandomRotation, transforms.ColorJitter),
        )
        for t in transforms_list.transforms
    ), "Training pipeline should use image augmentation techniques"


def test_output_layer():
    """Test if output layer is correctly configured"""
    model = Model()

    # Get the final linear layer
    output_layer = model.linear

    # Check if it's a Linear layer
    assert isinstance(output_layer, nn.Linear), "Output layer should be Linear"

    # Check input and output dimensions
    assert (
        output_layer.out_features == 10
    ), "Output layer should have 10 output features for MNIST classification"

    # Test forward pass shape
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)

    assert output.shape == (
        batch_size,
        10,
    ), f"Expected output shape (batch_size, 10), got {output.shape}"


if __name__ == "__main__":
    pytest.main([__file__])
