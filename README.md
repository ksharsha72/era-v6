# ERA-V6 Assignment

![Model Tests](https://github.com/ksharsha72/era-v6/actions/workflows/model-tests.yml/badge.svg)

## Description

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification with the following specifications:

- Parameters less than 20,000
- Uses Batch Normalization
- Uses Dropout
- Uses Image Augmentation techniques
- Achieves 99.4% validation accuracy

## Model Architecture

The model architecture includes:
- Multiple convolutional layers with batch normalization
- Dropout for regularization
- Image augmentation (RandomCrop, RandomRotation, ColorJitter)
- Total parameters: 9,768

## Tests

The repository includes automated tests that verify:
- Model has less than 20,000 parameters
- Batch normalization is implemented
- Image augmentation is configured correctly
- Output layer is properly configured

To run the tests: