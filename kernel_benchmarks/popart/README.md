# Kernel benchmarks: Convolutional and LSTM layers with PopART

This README describes how to run benchmarks for models with a single type of layer and synthetic data in training and inference.

## Overview

Each program creates a model with only one type of layer for benchmarking.
* LSTM (Long Short-Term Memory) are used in sequential data with long dependencies.
* Convolutional layers are used in image and video recognition, recommender systems, image classification, medical image analysis, natural language processing, and financial time series


## Running the model

This repo contains the code required to run the kernel model.

The structure of the repo is as follows:

| File                                            | Description			                                                   |
| ----------------------------------------------- | ---------------------------------------------------------              |
| `lstm.py`                                       | Benchmark program for 1 LSTM layer                                     |
| `conv.py`                                       | Benchmark program for 1 2D Convolutional layer                         |
| `README.md`                                     | This file                                                              |
| `tests/`                                        | Test code that can be run via pytest                                   |
| `requirements.txt`                              | Required packages to install                                           |

## Quick start guide

1. Prepare the environment. Install the Poplar SDK following the instructions
   in the Getting Started guide for your IPU system.
2. Run the training program. For example:

   `python3 lstm.py`
   
   Use `--help` to show all available options.

### Profiling

Profiling tools included in the Poplar SDK can be used to generate and view reports containing
profiling information for compilation and execution of the benchmarks.

The following enables report generation, and specifies a directory to generate reports in.

```
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"."}'
```

After running a benchmark, the resulting report can be opened using the PopVision Graph Analyser. 
See the Graph Analyser user guide for more information:
[PopVision Graph Analyser User Guide](https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/).
