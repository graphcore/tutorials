<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->

# Simple applications

This directory contains a number of basic applications written in different frameworks targeting the IPU.

## Poplar

- [Simple MNIST training example](poplar/mnist): this example shows how to build a training model to classify digits from the MNIST dataset.

## TensorFlow 2

- [Simple MNIST training example](tensorflow2/mnist): This example trains a simple 2-layer fully connected model on the MNIST numeral data set.

## PyTorch

### Complete examples of models

- [Classifying hand-written digits](pytorch/mnist) from the MNIST dataset is a well-known example of a basic machine learning task.

### Pre-trained models

- [Hugging Face's BERT](pytorch/bert) is a pre-trained BERT model made available by Hugging Face and which is implemented in PyTorch. This example consists of running one of the pre-trained BERT model on an IPU for an inference session.

## PopART

- [Simple MNIST Examples](popart/mnist): Contains 2 simple models, 1 linear and 1 using convolutions trained on the MNIST dataset.
