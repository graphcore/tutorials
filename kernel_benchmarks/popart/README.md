# Synthetic benchmarks on IPUs

This readme describes how to run synthetic benchmarks for models with a single type of layer and synthetic data in training and inference.

## Overview

#### LSTM Layer

This example uses an LSTM model for benchmarking. LSTM (Long Short-Term Memory) is used in sequential data with long dependencies.

#### 2D Convolutional Layer

This example uses a Convolutional layer for benchmarking. Convolutional layer employs a mathematical operation called convolution, and is used in image and video recognition, recommender systems, image classification, medical image analysis, natural language processing, and financial time series.

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
See the PopVision user guide for more information:
[PopVision User Guide](https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/index.html).


## File structure

|            |                           |
|------------|---------------------------|
| `lstm.py`          | Benchmark program for 1 LSTM layer                       |
| `conv.py`          | Benchmark program for 1 2D Convolutional layer           |


----

