<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# PopART Feature Examples

This directory contains a number of feature examples demonstrating how to use the Poplar Advanced Runtime (PopART). These include examples of how to use multiple IPUs and implement custom ops in Poplar, as well as other key features provided by PopART.


## Contents


### Multi IPU Examples

- [Sharding a Model over Multiple IPUs](sharding): This demo shows how to "shard" (split) a model over multiple IPUs using PopART.

- [Pipelining a Model over Multiple IPUs](pipelining): This demo shows how to use pipelining in PopART on a very simple model consisting of two dense layers.

- [Utilising Streaming Memory with Phased Execution](phased_execution): This example runs a network in inference mode over two IPUs by splitting it in several execution phases and keeping the weights in Streaming Memory.


### Further Examples

- [Custom Operators](custom_operators): This directory contains two example implementations of custom operators for PopART (Cube and LeakyReLU). Both examples create an operation definition with forward and backward parts, and include a simple inference script to demonstrate using the operators.

- [Data Callbacks](callbacks): This example creates a simple computation graph and uses callbacks to feed data and retrieve the results. Time between host-device transfer and receipt of the result on the host is computed and displayed for a range of different data sizes.

- [Automatic and Manual Recomputing](recomputing): This example shows how to use manual and automatic recomputation in popART with a seven layer DNN and generated data.
