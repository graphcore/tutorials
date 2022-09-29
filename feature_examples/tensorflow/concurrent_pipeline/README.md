<!-- Copyright (c) 2022 Graphcore Ltd. All rights reserved. -->
# Graphcore: Sharding Pipeline Stages using Concurrent Pipelines

## Concurrent Pipeline Support in TensorFlow

TensorFlow on IPU supports [concurrent
pipelines](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/3.0.0/tensorflow/perf_training.html#concurrent-pipeline-stages).
These are single pipeline stages that use more than a single IPU to allow tensor
parallel (sharded) computations to be defined in that stage. This code example
shows how to use this feature to implement a tensor parallel tied embedding
where the embedding lookup, projection, and final softmax operations are sharded
across multiple IPUs. It also contains a sharded ops library in `sharded.py`
that can be used to build other applications and an MNIST example showing how to
use the library in such an application.

A more general introduction to Pipelining can be found in our [memory and
performance optimisation
guide](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/3.0.0/optimising-performance.html#pipeline-execution-scheme).

### File structure

* `custom_ops` A TensorFlow Python module for accessing IPU specific sharded ops.
* `test` Test scripts and tools.
* `run_sharded_mnist.py` Simple example of pipelined MNIST that uses a sharded matmul in its final dense layer.
* `README.md` This file.

### How to use this example

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the [Getting Started
   guide for your IPU
   system](https://docs.graphcore.ai/en/latest/getting-started.html). Make sure
   to source the `enable.sh` script for Poplar and activate a Python virtualenv
   with a TensorFlow 1 wheel from the Poplar SDK installed (use the version
   appropriate to your operating system and processor).

2) Install required pip modules:

```bash
pip install -r requirements.txt
```

3) Build the custom ops.

```bash
make -j10
```

4) Run the Python code. The command below runs a simple test of a tied embedding
pipeline which checks the loss and embedding matrix gradient matches a JAX based
CPU implementation.

```bash
python3 tests/sharded_embedding_tool.py --ipus 4 --vocab-size 8000 --feature-size 768 --sequence-length 256
```

If you have previously built this module using a different SDK version you must
run `make clean` before re-running `make`.

## Concurrent Pipelined MNIST example

As well as the tests there is also an example of using the feature to train
MNIST in `run_sharded_mnist.py`. This can be used to analyse how the library
behaves in a training pipeline (the unit tests only check loss and gradients).
The program allows you to compare a standard pipeline with the concurrent one.
If you run the following commands you should see that they both train similarly
(same loss profile):

```bash
python3 run_sharded_mnist.py --pipeline-mode basic
python3 run_sharded_mnist.py --pipeline-mode concurrent
```

The following is a schematic representation of the basic MNIST pipeline model
in this example which splits layers across two IPUs:

```text
-------------------------- Basic Pipeline -----------------------------
IPU0: inputs -> MLP \
IPU1:                |-> Classifier -- SoftmaxCE -- Loss
```

However, in the concurrent pipeline case the final matrix multiply (classifier
layer) and the following softmax are executed tensor parallel in concurrent
stages:

```text
---------------------------- Concurrent Pipeline --------------------------------
IPU0: inputs -> MLP \ -> Classifier (top rows) -- SoftmaxCE(top)        |-> Combined Loss
IPU1:                |-> Classifier (bottom rows) -- SoftmaxCE(bottom) /
```
