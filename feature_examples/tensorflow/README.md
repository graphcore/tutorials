# Feature Examples for TensorFlow 1

This directory contains a number of feature examples for how to use TensorFlow 1 with IPUs. These include examples of how to measure performance, use multiple IPUs and implement custom ops in Poplar, among other things. This README provides an overview of all of the TensorFlow 1 examples.


## Performance

- [Inspecting tensors](inspecting_tensors): This example trains simple pipelined and non-pipelined models on the MNIST numeral data set and shows how tensors (containing activations and gradients) can be returned to the host for inspection using outfeed queues.
This can be useful for debugging a model.

- [I/O benchmarking](../simple_applications/tensorflow/mnist): The MNIST simple application shows how to use ipu.dataset_benchmark to determine the maximum achievable throughput for a given dataset.

## Using multiple IPUs

Simple examples demonstrating and explaining different ways of using multiple IPUs are provided. [Pipelining](pipelining) and [replication](replication) are generally used to parallelise and speed up training, whereas [sharding](sharding) is generally used to simply fit a model in memory.

- [PopDist training example](popdist): This shows how the PopDist API can be used to enable distributed training.

## Custom ops

- [Custom op example](custom_op): Code that demonstrates how to define your own custom op using Poplar and PopLibs and use it in TensorFlow 1.

- [Custom op example with gradient](custom_gradient): Code that demonstrates how to define your own custom op using Poplar and PopLibs and use it in TensorFlow 1. Also shows how to define the gradient of your custom op so that you can use automatic differentiation and operations that depend on it, such as the `minimize` method of an optimizer.


## Other examples

- [IPUEstimator](ipuestimator): Example of using the IPU implementation of the TensorFlow Estimator API.

- [Configuring IPU connections](connection_type): A code example which demonstrates how to use `ipu.utils.set_ipu_connection_type` to control if and when the IPU device is acquired.
