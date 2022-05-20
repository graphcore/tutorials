"""
Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
"""
# Training a simple TensorFlow 1 model on MNIST with an IPU

This tutorial shows how to train a simple model on the MNIST numerical
dataset on a single IPU. The dataset consists of 60,000 images of handwritten
digits (0-9) that must be classified according to which digit they represent.
"""
"""
We will do the following steps in order:

1. Load and pre-process the MNIST dataset from Keras.
2. Define a simple model.
3. Define and compile the training loop.
4. Configure the IPU system.
5. Train the model on the IPU.
"""
"""
## 1. Preparing your environment

In order to run this tutorial on the IPU you will need to have:

- A Poplar SDK environment enabled (see the
[Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide for your IPU system).
- The Graphcore port of TensorFlow 1 set up for the IPU (see the
[Setup Instructions](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html#setting-up-tensorflow-for-the-ipu))
"""
"""
To run the Jupyter notebook version of this tutorial:

1. Enable a Poplar SDK environment
2. In the same environment, install the Jupyter notebook server:
`python -m pip install jupyter`
3. Launch a Jupyter Server on a specific port:
`jupyter-notebook --no-browser --port <port number>`
4. Connect via SSH to your remote machine, forwarding your chosen port:
`ssh -NL <port number>:localhost:<port number> <your username>@<remote machine>`

For more details about this process, or if you need help troubleshooting,
see our [guide on using IPUs from Jupyter notebooks](../../../tutorials/standard_tools/using_jupyter/README.md).
"""
"""
## 2. Import necessary libraries

First of all, we need to import the Python modules that will be used in the example.
"""
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
import time

tf.disable_eager_execution()
tf.disable_v2_behavior()
# sst_hide_output
"""
## 3. Define the hyperparameters

We also need to specify the hyperparameters, which will be used later.
"""

BATCHSIZE = 32
EPOCHS = 5

"""
## 4. Prepare dataset

We can access the MNIST dataset through Keras.
"""

mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()

"""
The features are normalised by dividing each element of `x_train` (pixel values)
by 255. This will make our model converge faster.

We cast the labels to `int32` because other integer types are not, in general,
supported on the IPU.
"""

# Cast and normalize the training data
x_train = x_train.astype('float32') / 255
y_train = y_train.astype('int32')

"""
We create a `tf.data.Dataset` object from the data. When batching the data, we set
the `drop_remainder` to `True` so that all of our batches are guaranteed to have the
same number of examples. This is important because the IPU's Poplar software stack
does not support using tensors with shapes which are unknown when the program is
compiled.
"""

# Build iterator over the data
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().batch(BATCHSIZE, drop_remainder=True)
dataset_iterator = tf.data.make_initializable_iterator(dataset)

"""
## 5. Define the model

Next, we define a simple fully-connected network model using the standard Keras
Sequential API and create an instance of the model.
"""


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])
    return model

model = create_model()

"""
## 6. Define the loop body

Now that we have the dataset and the model, we need to define a function which executes
the main training loop.

Our function outputs the loss at each step so that we can track the performance of the model.
Because TensorFlow 1 uses lazy evaluation, we return `train_op` as well to ensure the training
step is executed.
"""


def training_loop_body(x, y):

    logits = model(x, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)

    return([loss, train_op])

"""
## 7. Prepare the model for the IPU

Now we can build an executable TensorFlow operation from the loop function, which is handled
by `ipu.ipu_compiler.compile`. This takes a Python function computation and a list of inputs
`inputs` and returns an operation which applies the computation to the inputs and can be run
on an IPU device using a `sess.run` call. These inputs can be constants, `tf.placeholder`
variables, or values from a dataset iterator. If we wish to pass inputs from a dataset iterator,
we pass them from the `get_next()` method of the iterator.

Note that we build the operation within the scope of a particular device with `ipu.scope.ipu_scope()` API.
"""

# Get inputs from get_next() method of iterator
(x, y) = dataset_iterator.get_next()

with ipu.scopes.ipu_scope('/device:IPU:0'):

    training_loop_body_on_ipu = ipu.ipu_compiler.compile(computation=training_loop_body, inputs=[x, y])
# sst_hide_output
"""
## 8. Add IPU configuration

To use the IPU, we must create an IPU configuration.
We can use `cfg.auto_select_ipus = 1` to automatically select one IPU.
"""

ipu_configuration = ipu.config.IPUConfig()
ipu_configuration.auto_select_ipus = 1
ipu_configuration.configure_ipu_system()

"""
## 9. Execute in a TF session

We can now run our training loop on an IPU using a TensorFlow session, with no further
IPU-specific code required.
"""

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(dataset_iterator.initializer)

    batches_per_epoch = len(x_train)//BATCHSIZE

    for epoch in range(EPOCHS):

        loss_running_total = 0.0

        epoch_start_time = time.time()

        for batch in range(batches_per_epoch):

            # This part runs on IPU since train_loop_body
            # is placed under ipu_scope
            loss = sess.run(training_loop_body_on_ipu)

            loss_running_total += loss[0]

        # Print average loss and time taken for epoch
        print('\n', end='')
        print("Loss:", loss_running_total/batches_per_epoch)
        print("Time:", time.time() - epoch_start_time)

print("Program ran successfully")
# sst_hide_output
"""
## Other useful resources

- [TensorFlow Docs](https://docs.graphcore.ai/en/latest/software.html#tensorflow):
all Graphcore documentation specifically relating to TensorFlow.

- [IPU TensorFlow 1 Code Examples](https://github.com/graphcore/examples/tree/master/code_examples/tensorflow):
examples of different use cases of TensorFlow 1 on the IPU.

- [Graphcore tutorials](https://github.com/graphcore/tutorials/tree/master/tutorials):
a list of existing tutorials for using the IPU.
"""
