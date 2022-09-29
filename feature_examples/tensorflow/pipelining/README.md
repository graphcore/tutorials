<!-- Copyright (c) 2020 Graphcore Ltd. All rights reserved. -->
# TensorFlow Pipelining example

This example shows how to use pipelining in TensorFlow to train a very simple model
consisting of just dense layers.

## File structure

* `pipelining.py` The main TensorFlow file showcasing pipelining.
* `README.md` This file.
* `requirements.txt` Required modules for testing.
* `test_pipelining.py` Script for testing this example.

## How to use this example

1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to source the `enable.sh` script for Poplar and activate a Python virtualenv with
   a TensorFlow 1 wheel from the Poplar SDK installed (use the version appropriate to your operating system and processor).

2) Run the script.

    `python pipelining.py`

3) To profile this example you should use the Poplar environment variable below and run it for a single step.

    `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true"}' python pipelining.py --run-single-step`

    This will produce a report directory that starts with "tf_report_" and is postfixed with a timestamp.

#### Options

Run pipelining.py with the -h option to list all the command line options.

### Tests

1) Install the requirements.

    `python3 -m pip install -r requirements.txt`

2) Run the tests.

    `python3 -m pytest test_pipelining.py`
