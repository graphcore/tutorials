<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# Using a custom operator in a PyTorch model

This example shows how to use a custom operator in the PopTorch
framework on the IPU.

To be used in PopTorch, a custom operator must first be implemented as
a PopART custom operator, then be made available to PopTorch. It is this
last stage that is shown in this example, using the custom operator created
in our [PopART custom Leaky ReLU operator example](../../popart/custom_operators/leaky_relu_example).

This example shows the process of loading in a custom operator and using it in a simple
model creation and training process. This is shown with a CNN using the LeakyReLU custom
operator as an activation function, on the FashionMNIST dataset.

For more information on custom operators in PopTorch, please refer to the
[Creating custom operators section of our PyTorch for the IPU User Guide](https://docs.graphcore.ai/projects/poptorch-user-guide/en/3.0.0/overview.html#creating-custom-ops).

## File structure

* `Makefile` Simple makefile which builds the Poplar code
* `requirements.txt` Required packages to run the Python file.
* `tests/test_poptorch_custom_op.py` Script for testing this example.
* `poptorch_custom_op.py` PopTorch CNN program using the custom operator.

In the [PopART custom Leaky ReLU operator example](../../popart/custom_operators/leaky_relu_example):
* `leaky_relu_custom_op.cpp` Custom code which defines and generates the custom operator.

## How to run the example

1) Prepare the environment:
    - Ensure the Poplar SDK is installed (follow the instructions in the Getting
    Started guide for your IPU system: <https://docs.graphcore.ai/en/latest/getting-started.html>.
    - Install the requirements for the Python program with:
       ```
       python3 -m pip install -r requirements.txt
       ```
2) Build the custom operator in the [PopART Leaky ReLU example](../../popart/custom_operators/leaky_relu_example) (after making sure that the `Makefile` and `leaky_relu_custom_op.cpp` files are present):
      ```
      cd ../../popart/custom_operators/leaky_relu_example
      make
      cd -
      ```
3) Run the Python example:
    ```
    python3 poptorch_custom_op.py
    ```
