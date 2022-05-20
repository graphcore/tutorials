# Graphcore tutorials

These tutorials provide hands-on programming exercises to enable you to familiarise yourself with creating, running and profiling programs on the IPU. They are part of the Developer resources provided by Graphcore: https://www.graphcore.ai/developer.

Each of the tutorials contains its own README file with full instructions.

## Poplar

- [Tutorial 1: Programs and variables](poplar/tut1_variables)
- [Tutorial 2: Using PopLibs](poplar/tut2_operations)
- [Tutorial 3: Writing vertex code](poplar/tut3_vertices)
- [Tutorial 4: Profiling output](poplar/tut4_profiling)
- [Tutorial 5: Matrix-vector multiplication](poplar/tut5_matrix_vector)
- [Tutorial 6: Matrix-vector multiplication optimisation](poplar/tut6_matrix_vector_opt)

## TensorFlow 1

Getting started with the IPU:
- [Starter tutorial: MNIST training example](../simple_applications/tensorflow/mnist)
- [Tutorial 1: Porting a simple example](tensorflow1/basics/tut1_porting_a_model)
- [Tutorial 2: Loops and data pipelines](tensorflow1/basics/tut2_loops_data_pipeline)
- [Training a model using half- and mixed-precision](tensorflow1/half_precision_training)

Using multiple IPUs and handling large models:
- [Converting a model to run on multiple IPUs with pipelining](tensorflow1/pipelining)

## TensorFlow 2

Getting started with the IPU:
- [Starter tutorial: MNIST training example](../simple_applications/tensorflow2/mnist)
- [TensorFlow 2 Keras: How to run on the IPU](tensorflow2/keras)

Exchanging data between the host and the IPU:
- [TensorFlow 2: How to use infeed/outfeed queues](tensorflow2/infeed_outfeed)

Debugging and analysis:
- [TensorFlow 2: How to use TensorBoard](tensorflow2/tensorboard)

## PyTorch

Getting started with the IPU:
- [Starter tutorial: MNIST training example](../simple_applications/pytorch/mnist)
- [From 0 to 1: Introduction to PopTorch](pytorch/tut1_basics)

Exchanging data between the host and the IPU:
- [Efficient data loading with PopTorch](pytorch/tut2_efficient_data_loading)

Maximising compute on the IPU:
- [Half and mixed precision in PopTorch](pytorch/tut3_mixed_precision)

Using multiple IPUs and handling large models:
- [Parallel execution using pipelining](pytorch/tut5_pipelining)
- [Fine-tuning BERT with HuggingFace and PopTorch](pytorch/tut_finetuning_bert)

Debugging and analysis:
- [Observing tensors in PopTorch](pytorch/tut4_observing_tensors)

## PopVision

PopVision is Graphcore's suite of graphical application analysis tools.

- [Tutorial 1: Instrumenting applications and using the PopVision System Analyser](popvision/tut1_system_analyser_instrumentation)
- [Tutorial 2: Accessing profiling information with libpva](popvision/tut2_libpva)
- Profiling output with the PopVision Graph Analyser is currently included in [Poplar Tutorial 4: profiling output](poplar/tut4_profiling)

## Standard tools

In this folder you will find explanations of how to use standard deep learning tools
with the Graphcore IPU. Guides included are:

- [Using IPUs from Jupyter Notebooks](standard_tools/using_jupyter)
