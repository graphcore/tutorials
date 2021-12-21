# Graphcore feature examples

The code examples demonstrate features which will enable you to make the most of the IPU. They are part of the Developer resources provided by Graphcore: https://www.graphcore.ai/developer.

Each of the examples contains its own README file with full instructions.


## PopART

Efficiently use multiple IPUs and handle large models:
- [Phased execution](popart/phased_execution): this example shows how to run a network over two IPUs by splitting it in
several execution phases.
- [Pipelining](popart/pipelining): a simple model made of two dense layers, pipelined over two IPUs.
- [Recomputing](popart/recomputing): a demonstration of manual and automatic recomputing on the IPU.
- [Sharding](popart/sharding): a simple model sharded on two IPUs.

Exchange data between host and IPU efficiently:
- [Callbacks](popart/callbacks): a simple computation graph that uses callbacks to feed data and
retrieve the results.

Define custom operators:
- [Custom operators](popart/custom_operators): two implementations of custom operators 
([leaky ReLU](popart/custom_operators/leaky_relu_example) and [cube](popart/custom_operators/cube_op_example)).

## Poplar

Exchange data between host and IPU efficiently:
- [Prefetch](poplar/prefetch): a demonstration of prefetching data when a program runs several times.

Demonstrate advanced features of Poplar:
- [Advanced example](poplar/advanced_example): an example demonstrating several advanced features of Poplar, including
saving and restoring Poplar executables, moving I/O into separate Poplar programs, and using our PopLibs framework.


## TensorFlow 1

Debugging and analysis:
- [Inspecting tensors](tensorflow/inspecting_tensors): an example that shows how outfeed queues can be used to
return activation and gradient tensors to the host for inspection.

Efficiently use multiple IPUs and handle large models:
- [Pipelining](tensorflow/pipelining): a simple model made of two dense layers, pipelined over two IPUs.
- [PopDist](tensorflow/popdist): an example showing how to make an application ready for distributed training and inference by using the PopDist
library, and how to launch it with the PopRun distributed launcher.
- [Replication](tensorflow/replication): an example showing how to use replication in TensorFlow to train
a simple CIFAR-10 convolution model.
- [Sharding](tensorflow/sharding): a simple model sharded on two IPUs.

Use estimators:
- [IPU Estimator](tensorflow/ipuestimator): an example showing how to use the IPUEstimator to train and evaluate
a simple CNN.

Control IPU use:
- [Connection Type](tensorflow/connection_type): a demonstration of controlling if and when an IPU device is acquired using the `device_connection.type` configuration option.

Define custom operators:
- [Custom operator](tensorflow/custom_op): a simple custom operator that adds two vectors of arbitrary size, created in Poplar
and used in a TensorFlow model.
- [Custom gradient](tensorflow/custom_gradient): a custom operator for the batched dot product, defining both the forward
operator and its gradient in Poplar, then used in a TensorFlow model.


## TensorFlow 2

Debugging and analysis:
- [Inspecting tensors](tensorflow2/inspecting_tensors): an example that shows how outfeed queues can be used to
return activation and gradient tensors to the host for inspection.

Efficiently use multiple IPUs and handle large models:
- [PopDist](tensorflow/popdist): an example showing how to make an application ready for distributed training and inference by using the PopDist
library, and how to launch it with the PopRun distributed launcher.

Use estimators:
- [IPU Estimator](tensorflow2/ipu_estimator): an example showing how to use the IPUEstimator to train and evaluate
a simple CNN.

Specific layers:
- [Embeddings](tensorflow2/embeddings): an example of a model with an embedding layer and an
LSTM, trained on the IPU to predict the sentiment of an IMDB review.


## PyTorch

Efficiently use multiple IPUs and handle large models:
- [PopDist](pytorch/popdist): an example showing how to make an application ready for distributed training and inference by using the PopDist
library, and how to launch it with the PopRun distributed launcher.

Define custom operators:
- [Custom operators](pytorch/custom_op): an example showing how to make a PopART custom operator available
to PopTorch and how to use it in a model.

Specific layers:
- [Octconv](pytorch/octconv): an example showing how to use Octave Convolutions in PopTorch training and inference models.
