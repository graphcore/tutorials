# Kernel benchmarks: Grouped convolution, GRU, LSTM and RNN layers with TensorFlow

This README describes how to run benchmarks for models with a single type of layer and synthetic data in training and inference.

## Overview

Each program creates a model with only one type of layer for benchmarking.
* Grouped convolutions are used in CNNs such as ResNeXt models.
* RNN (Recurrent Neural Network) are used in sequential data.
* LSTM (Long Short-Term Memory) are used in sequential data with long dependencies.
* GRU (Gated Recurrent Unit) are a simpler version of LSTM.
* Dense layers, also known as fully connected layers are widely used across a range of models.
* HMC (Hamiltonian Monte Carlo) is a step from Contrastive Divergence for Combining Variational Inference and MCMC.

## Running the model

This repo contains the code required to run the kernel model.

The structure of the repo is as follows:

| File                                            | Description			                                                       |
| ----------------------------------------------- | ---------------------------------------------------------              |
| `grouped_conv.py`                               | Benchmark program for grouped convolutions                             |
| `gru.py`                                        | Benchmark program for 1 GRU layer                                      |
| `hmc.py`                                        | Benchmark HMC steps in isolation                                       |
| `lstm.py`                                       | Benchmark program for 1 LSTM layer                                     |
| `lstm_multi.py`                                 | Benchmark for a multi-layer LSTM with a dense final layer              |
| `rnn.py`                                        | Benchmark program for 1 RNN layer                                      |
| `dense.py`                                      | Benchmark program for 1 Dense layer                                    |
| `README.md`                                     | This file                                                              |
| `test/`                                         | Test code that can be run via pytest                                   |
| `requirements.txt`                              | Required packages to install                                           |

## Quick start guide

1. Prepare the environment. Install the Poplar SDK following the instructions
   in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` 
   script for Poplar and activate a Python virtualenv with the tensorflow-1 wheel
   from the Poplar SDK installed.
2. Run the training program. For example:

   `python3 program.py`

   Each benchmark can be executed multiple times using the `--steps`
   option.

   `python3 program.py --steps 5`
   
   Use `--help` or `-h` to show all available options.


### Profiling

Profiling tools included in the Poplar SDK can be used to generate and view reports containing
profiling information for compilation and execution of the benchmarks.

The following enables report generation, and specifies a directory to generate reports in.

```
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"."}'
```

After running a benchmark, the resulting report can be opened using the PopVision Graph Analyser. 
See the PopVision user guide for more information:
[PopVision User Guide](https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/index.html).

### License

The Hamiltonian Monte Carlo benchmark (`hmc.py`) is licensed under the Apache License 2.0 - see the LICENSE file in this directory.

This example contains work derived from the following: [TensorFlow `tfp.mcmc.HamiltonianMonteCarlo` documentation](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo).

All other examples in this directory are licensed under the MIT license - see the LICENSE file at the root of the repository.

Copyright (c) 2021 Graphcore Ltd. All rights reserved.