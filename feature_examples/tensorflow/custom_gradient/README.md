<!-- Copyright (c) 2020 Graphcore Ltd. All rights reserved. -->
# Creating a TensorFlow Custom Operator with Gradient

Creates a custom operator (a batched dot product) defining both the
forward operator and its gradient in Poplar code. Uses the custom operator
in a simple logistic regression optimisation program which checks
the results with the custom operator match those from the built-in operator.

## File structure

* `Makefile` Simple Makefile that builds the Poplar shared object.
* `product.cpp` Poplar code that describes the forward and grad operators.
* `regression.py` TensorFlow program that uses the custom operator to do logistic regression.
* `requirements.txt` Required packages.
* `test_regression.py` Script for testing this example.
* `README.md` This file.

## How to use this example

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to source the `enable.sh` script for Poplar.

2) Build the custom operator and then run the Python code:

```
make
python3 regression.py
```
