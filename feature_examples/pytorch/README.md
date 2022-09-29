<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# Code Examples for PyTorch

This directory contains code examples showing how to use PyTorch with Graphcore's IPUs. These include full models as well as examples of how to use pre-trained models.

## Complete examples of models

- [Octave Convolutions](octconv) are a novel convolutional layer in neural networks. This example shows an implementation of how to train the model and run it for inference.

## Using multiple IPUs

- [PopDist training example](popdist): This shows how to make an application ready for distributed training by using the PopDist API, and how to launch it with the PopRun distributed launcher.

## Custom operators

- [Custom Operators](custom_op) shows the use of a custom operator in PopTorch. This example shows an implementation of the LeakyReLU custom operator in the training of a simple model.
