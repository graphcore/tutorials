# Using a custom op in a PyTorch model

This example describes the use of a custom op created in Poplar for the PopTorch
framework on the IPU. It does not describe the *creation* of a custom op with
Poplar in C++, only the use of the op when developing a PyTorch model.

To see what a created custom op looks like, see [this example](tutorials/feature_examples/popart/custom_operators/leaky_relu_example) of
the Leaky ReLU custom op.

This example shows the process of loading in a custom op and using it in a simple
model creation and training process. This is shown with a CNN using the LeakyReLU custom
op as an activation function, on the FashionMNIST dataset. 

# File structure

* `leaky_relu_custom_op.cpp` Custom code which defines and generates the custom op.
* `Makefile` Simple makefile which builds the Poplar code
* `requirements.txt` Required packages to run the Python file.
* `test_poptorch_custom_op.py` Script for testing this example.
* `poptorch_custom_op.py` PopTorch CNN program using the custom op.

# Using the example

1) Prepare the environment:
    - Ensure the Poplar SDK is installed (follow the instructions in the Getting
    Started guide for your IPU system: https://docs.graphcore.ai/en/latest/getting-started.html).
    - Install the requirements for the Python program with:
       ```pip3 install -r requirements.txt```
    - Build the custom op in `tutorials/feature_examples/popart/custom_operators/leaky_relu_example`,
      (Ensure the `Makefile` and `leaky_relu_custom_op.cpp` files are present):
      ```cd ../../popart/custom_operators/leaky_relu_example```
      ```make```
      ```cd -```
    - Run the Python example: 
      ```python3 poptorch_custom_op.py```