# Inspecting tensors using outfeed queues and a custom optimizer

In this example we train simple pipelined and non-pipelined models on the MNIST numeral dataset
and show how outfeed queues can be used to return activation and gradient
tensors to the host for inspection.

This can be useful for debugging but will significantly increase the amount
of memory required on the IPUs. A small value for the gradient accumulation count
should be used to mitigate this.
Consider executing the model for a small number of times in each step.
Filters can be used to only return a subset of the activations and gradients.

This example will run in TensorFlow 1 or TensorFlow 2.

## File structure

This example consists of four source files:

* `maybe_outfeed_queue.py` Contains the `MaybeOutfeedQueue` class - a wrapper for an
  IPUOutfeedQueue that allows key-value pairs to be selectively added to a dictionary
  that can then be enqueued. A list of filters can be supplied to the constructor.
* `outfeed_optimizer.py` Contains the `OutfeedOptimizer` class - a custom optimizer
  that enqueues gradients using a `MaybeOutfeedQueue`,
  with the choice of whether to enqueue the gradients after they are computed
  (the pre-accumulated gradients) or before they are applied (the accumulated gradients).
* `pipelined_model.py` A Python script which contains a simple pipelined model
  that will selectively outfeed some of the activations and gradients.
  The pipeline stages use the `MaybeOutfeedQueue` to outfeed activations.
* `model.py` A Python script containing a non-pipelined model with the same
  functionality.

See the source code for further documentation.

The directory also includes:

* `README.md` This file.
* `tests` Subdirectory containing test scripts.


## How to use this example

1) Prepare the TensorFlow environment

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to source the `enable.sh` script for Poplar and activate a Python virtualenv with either
   a TensorFlow 1 or TensorFlow 2 wheel from the Poplar SDK installed
   (use the version appropriate to your operating system).

2) Run one of the model scripts

    `python pipelined_model.py`

Example output:

```
Activation key: dense1 shape: (1600, 32, 256)
Activation key: dense3 shape: (1600, 32, 10)

Step 0 - Summary Stats
Index Name        Mean         Std          Minimum      Maximum      NaNs    infs
0     dense1      0.321088     0.425143     0.000000     4.763036     False   False
1     dense3      -0.055209    2.410968     -11.534188   15.033943    False   False

Gradient key: dense2/bias:0_grad shape: (100, 128)
Gradient key: dense2/kernel:0_grad shape: (100, 256, 128)
Gradient key: dense1/bias:0_grad shape: (100, 256)
Gradient key: dense1/kernel:0_grad shape: (100, 784, 256)

Step 0 - Summary Stats
Index Name                      Mean         Std          Minimum      Maximum      NaNs    infs
0     dense2/bias:0_grad        -0.010146    0.110202     -0.727358    0.696234     False   False
1     dense2/kernel:0_grad      -0.002280    0.051605     -1.077855    1.028231     False   False
2     dense1/bias:0_grad        -0.009367    0.087506     -0.668157    0.686126     False   False
3     dense1/kernel:0_grad      -0.000908    0.023315     -0.447516    0.434358     False   False

Epoch 0.0, Mean loss: 0.699
```

#### Options

The following command line options are available. See the code in `pipelined_model.py`
and `model.py` for other ways of changing the behaviour of the example, such as changing the
filters used by the `MaybeOutfeedQueue` objects.

* `--outfeed-pre-accumulated-gradients`: If set then outfeed the pre-accumulated
   rather than accumulated gradients.
* `--gradient-accumulation-count`: Integer (default is 16).
   The number of mini-batches for which gradients are accumulated before performing
   a weight update. When pipelining this is the number of times each pipeline stage
   will be executed in each pipeline execution.
* `--repeat-count`: Integer (default is 100). The number of times the model will be
   executed for each step. For `model.py` this must be a multiple of
   `--gradient-accumulation-count`.
* `--epochs`: Integer (default is 3). Total number of epochs to train for.
* `--run-single-step`: Runs for a single step to minimise the size of execution profile data.

An example command line to use for profiling is:

```
   POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true"}' python pipelined_model.py --gradient-accumulation-count 4 --repeat-count 1 --run-single-step
```
