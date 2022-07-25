# Feature examples for TensorFlow 2

This directory contains several examples showing how to use TensorFlow 2 on the IPU.

- [CIFAR-10 with IPUEstimator](ipu_estimator): This example shows how to train a model to sort images from the CIFAR-10 dataset using the IPU implementation of the TensorFlow Estimator API.

- [IMDB Sentiment Prediction](embeddings): These examples train an IPU model with an embedding layer and an LSTM to predict the sentiment of an IMDB review.

- [Inspecting tensors using custom outfeed layers and a custom optimizer](inspecting_tensors): This example trains a choice of simple fully connected models on the MNIST numeral data set and shows how tensors (containing activations and gradients) can be returned to the host via outfeeds for inspection.

- [PopDist training and inference example](popdist): This shows how to make an application ready for distributed training and inference by using the PopDist API, and how to launch it with the PopRun distributed launcher.

- [Recomputation Checkpoints](recomputation_checkpoints): This example demonstrates using checkpointing of intermediate values to reduce live memory peaks with a simple Keras LSTM model.