# MNIST Image Classification with ResNet and HybridModel
Team Members: Manohar kota, Safia shaik, Bhagya teja chalicham

This repository contains implementations of neural network model for classifying the MNIST dataset:

1. ResNet (pretrained on ImageNet, modified for MNIST)
2. HybridModel (a custom hybrid network combining a simple ResNet-like backbone with a fully connected classifier)


# Table of Contents
* Project Overview
* Environment Setup
* Running the Code
* File Structure
* Model Details
* License

# Project Overview

This project demonstrates how to:

* Use a pre-trained ResNet model to classify MNIST digits with minimal modification.
* Build a HybridModel with a custom, simplified ResNet backbone and a classifier head.
* This model aim to classify images from the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9).

# Environment Setup
To run this code, you need to have Python 3.x and the following dependencies installed. It is recommended to use a virtual environment to manage dependencies.

# Required Python Version
Python 3.x (Recommended: Python 3.8 or newer)

# Dependencies
* PyTorch: For training and evaluating the models.
* torchvision: For the pre-trained ResNet18 model and dataset utilities.
* numpy: For numerical operations.
* matplotlib: For any optional plotting (not used directly in this code, but useful for visualizing training).

# Install Dependencies
First, set up a virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install torch torchvision numpy

# Running the Code
To run the training and evaluation, simply execute the Python script:

python train.py

This will:

* Load the MNIST dataset (training and testing sets).
* Train the ResNet18ForMNIST model (ResNet18 with some modifications for MNIST).
* Optionally, train the HybridModel model by modifying the code to include this model class.
* Output training statistics and test accuracy after the training loop.

# Training Process
* The model will be trained for 3 epochs by default.
* The training process outputs the loss and accuracy every 100 batches.
* After training, the model will be evaluated on the test set, and the best model (based on test accuracy) will be saved as best_resnet18_model.pth.

# Model Details
1. ResNet18ForMNIST (Pre-trained ResNet18 for MNIST)
* Architecture: Uses a pre-trained ResNet18 model from torchvision.models, with modifications to:
* Adapt the first convolutional layer to accept single-channel grayscale images (28x28 pixels).
* Change the final fully connected layer to output 10 classes (corresponding to the 10 MNIST digits).
2. HybridModel
* Architecture: A custom hybrid model with a simplified ResNet-like backbone.
* SimpleResNet: A custom module that uses convolutional layers with batch normalization, followed by a two-layer ResNet-style structure.
* Classifier: A fully connected classifier that flattens the output of the feature extractor and feeds it through a couple of linear layers to output 10 classes.
* The model is trained using CrossEntropyLoss and the Adam optimizer.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
