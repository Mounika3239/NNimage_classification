# Neural Network Implementation for Image Classification

## Overview
This project implements a Multilayer Perceptron (MLP) neural network to classify images from the MNIST Handwritten Digits and CelebA Face Dataset. The repository includes hyperparameter tuning, performance evaluation, and comparisons with Deep Neural Networks (DNN) and Convolutional Neural Networks (CNN).

## Datasets
- MNIST: 60,000 training images, 10,000 test images (28x28 grayscale).
- CelebA: 26,407 celebrity face images (54x44), classified as "with glasses" or "without glasses."

## Implementation Details
- Feedforward and Backpropagation.
- Regularization to avoid overfitting.
- Hyperparameter tuning (hidden units and lambda values).

## Performance Comparison:
- MNIST Dataset: MLP vs. DNN.
- CelebA Dataset: MLP vs. DNN vs. CNN.

## Tools Used:
Python 3
NumPy
SciPy
TensorFlow (for DNN and CNN comparisons).

## How to Run
1. Clone the repository:
- git clone https://github.com/Mounika3239/NNimage_classification.git
- cd NNimage_classification

2. Install dependencies:
- pip install numpy scipy tensorflow

3. Run the scripts:
*For MNIST:*
- python nnScript.py
*For CelebA:*
- python facennScript.py
  
4. Review the results:
- Optimal hyperparameters are stored in params.pickle



