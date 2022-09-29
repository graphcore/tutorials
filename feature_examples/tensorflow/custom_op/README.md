<!-- Copyright (c) 2020 Graphcore Ltd. All rights reserved. -->
# Creating a simple TensorFlow Custom Operator

Creates a simple custom operator that adds two vectors of arbitrary size. The operator
is created in Poplar using a custom vertex. This simple example does not show
how to create the corresponding gradient operator: to see how to create a custom gradient operator,
see the [TensorFlow Custom Gradient example](../custom_gradient).

## File structure

* `custom_codelet.cpp` Custom codelet used in the custom operator.
* `Makefile` Simple Makefile that builds the Poplar code and codelet (gp file).
* `poplar_code.cpp` Poplar code that builds the custom operator.
* `requirements.txt` Required packages.
* `tf_code.py` TensorFlow program that uses the custom operator.
* `test_custom_op.py` Script for testing this example.
* `README.md` This file.

## How to use this example

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to source the `enable.sh` script for Poplar.

2) Build the custom operator and then run the Python code.

```
make
python3 tf_code.py
```
