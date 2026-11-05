# Denoising Autoencoder on MNIST

## Overview

This project implements a **denoising autoencoder in PyTorch** to learn compact representations of handwritten digits from the MNIST dataset. The model is trained to reconstruct clean images from inputs corrupted with dropout noise. The notebook also visualizes reconstructions and latent-unit activations, then explores using the encoder for digit classification with and without autoencoder pretraining.

## Key Features

- Loads MNIST and creates training and validation splits (55,000 and 5,000 images).
- Builds an encoder–decoder network using fully connected layers and ReLU activations.
- Applies dropout to corrupt input images during denoising training.
- Trains with **NAdam** optimization and **mean squared error (MSE)** loss; tracks reconstruction RMSE.
- Visualizes original and reconstructed images.
- Displays examples that most strongly activate neurons in the 128-dimensional coding layer.
- Compares a classification head using the pretrained encoder with a randomly initialized encoder on a small labeled training subset.

## Model Architecture

**Encoder**
- Flatten 28 × 28 images into 784 features.
- Apply dropout with probability 0.5.
- Fully connected layers: `784 → 128 → 128`, with ReLU activations.

**Decoder**
- Fully connected layers: `128 → 128 → 784`.
- ReLU hidden activation and sigmoid output activation.
- Reshape the output to `1 × 28 × 28`.

The resulting model learns to reconstruct an image from a dropout-corrupted version of that image.

## Dataset

The notebook uses the MNIST training dataset from `torchvision.datasets.MNIST`. It splits the 60,000 training examples into:
- **Training:** 55,000 images
- **Validation:** 5,000 images

For autoencoder training, each image is used as both the input target pair (`image → image`). The notebook uses a batch size of 128.

## Training

- Framework: PyTorch
- Optimizer: NAdam (`lr=0.01`)
- Reconstruction loss: Mean Squared Error (MSE)
- Reconstruction metric: Root Mean Squared Error (RMSE)
- Epochs: 10
- Device: CUDA (`cuda` is specified in the notebook)

The notebook also includes a downstream classification experiment. It freezes the encoder and trains a 10-class linear output layer, comparing the pretrained encoder against a separately initialized encoder. The classification experiment uses a 500-example training subset and cross-entropy loss.

## Requirements

Install the core dependencies in an environment with a compatible PyTorch setup:

```bash
pip install torch torchvision torchmetrics matplotlib jupyter
```

A CUDA-capable GPU and compatible PyTorch/CUDA installation are needed to run the notebook as written, because tensors and models are explicitly moved to `cuda`. To run on a CPU-only machine, update those device assignments accordingly.

## Run the Notebook

1. Clone or download this repository.
2. Install the dependencies listed above.
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `denoising_autoencoder.ipynb` and run the cells in order.

The MNIST dataset is downloaded automatically by `torchvision` when it is not already present in `./data`.

## What You Can Explore

- How dropout noise affects image reconstruction.
- How an encoder compresses images into a lower-dimensional representation.
- Which input images strongly activate individual latent units.
- Whether autoencoder pretraining helps a downstream digit-classification model.
