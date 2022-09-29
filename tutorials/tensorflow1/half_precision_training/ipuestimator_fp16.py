# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Example of training a simple convolutional model on Fashion-MNIST using mixed precision
# In this example, we use the IPUEstimator class rather than running directly in a session

import time
import argparse

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

# Process command-line arguments

parser = argparse.ArgumentParser(
    description="Train a simple convolutional model on Fashion-MNIST"
)

parser.add_argument(
    "chosen_precision_str",
    metavar="precision",
    choices=["mixed", "float32"],
    type=str,
    help="Precision to use",
)

parser.add_argument(
    "--batch-size", type=int, default=32, help="Batch size to use for training"
)

parser.add_argument(
    "--epochs", type=int, default=5, help="Number of epochs to train for"
)

parser.add_argument(
    "--loss-scaling-factor",
    type=float,
    default=2**8,
    help="Scaling factor for loss scaling",
)

parser.add_argument(
    "--learning-rate", type=float, default=0.01, help="Learning rate for the optimizer"
)

parser.add_argument(
    "--use-float16-partials",
    action="store_true",
    help="Use FP16 partials in matmuls and convs",
)

args = parser.parse_args()

# Computations done in chosen precision, which is float16 for mixed precision
if args.chosen_precision_str == "mixed":
    compute_precision_str, compute_precision = "float16", tf.float16
else:
    compute_precision_str, compute_precision = "float32", tf.float32

# Load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), _ = fashion_mnist.load_data()

# Compute batches per epoch
# Use floor division because we drop the remainder
batches_per_epoch = len(x_train) // args.batch_size

# Normalize and cast the data
x_train = x_train.astype(compute_precision_str) / 255
y_train = y_train.astype("int32")


# FP32 parameter getter
# This function creates FP32 weights no matter what the compute dtype is


def fp32_parameter_getter(getter, name, dtype, trainable, shape=None, *args, **kwargs):

    if trainable and dtype != tf.float32:
        parameter_variable = getter(
            name, shape, tf.float32, *args, trainable=trainable, **kwargs
        )
        return tf.cast(parameter_variable, dtype=dtype, name=name + "_cast")

    else:
        parameter_variable = getter(
            name, shape, dtype, *args, trainable=trainable, **kwargs
        )
        return parameter_variable


# Define a convolution that uses tf.get_variable to create the kernel
# We use different `op_name`s for each operation so the variables are all given different names
def conv(feature_map, kernel_size, stride, filters_out, op_name, padding="SAME"):

    # We use NHWC format
    filters_in = feature_map.get_shape().as_list()[-1]

    # Resource variables must be used on the IPU
    with tf.variable_scope(op_name, use_resource=True):

        kernel = tf.get_variable(
            name="conv2d/kernel",
            shape=[kernel_size, kernel_size, filters_in, filters_out],
            dtype=feature_map.dtype,
            trainable=True,
        )

        return tf.nn.conv2d(
            feature_map,
            filters=kernel,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format="NHWC",
        )


# Define a dense layer that uses tf.get_variable to create the weights and biases
def dense(inputs, units_out, op_name):

    flattened_inputs = tf.layers.flatten(inputs)

    flattened_inputs_size = flattened_inputs.get_shape().as_list()[-1]

    # Expand dimensions to do batched matmul
    flattened_inputs = tf.expand_dims(flattened_inputs, -1)

    with tf.variable_scope(op_name, use_resource=True):

        weights = tf.get_variable(
            name="weights",
            shape=[units_out, flattened_inputs_size],
            dtype=inputs.dtype,
            trainable=True,
        )

        biases = tf.get_variable(
            name="biases", shape=[units_out, 1], dtype=inputs.dtype, trainable=True
        )

        return tf.matmul(weights, flattened_inputs) + biases


# Make IPUEstimatorSpec by defining training loop
# In this case, we are only interested in training
#     and we ignore the `params` argument
def make_estimator_spec(features, labels, mode, params):

    if mode != tf.estimator.ModeKeys.TRAIN:
        raise Exception("Only training is supported")

    # Apply the model function to the inputs, using
    #      the chosen variable getter as our custom getter
    with tf.variable_scope(
        "all_vars", use_resource=True, custom_getter=fp32_parameter_getter
    ):
        layer_out = tf.reshape(features, [args.batch_size, 28, 28, 1])

        layer_out = conv(
            layer_out, kernel_size=3, stride=1, filters_out=32, op_name="conv1"
        )

        layer_out = tf.nn.relu(layer_out)

        layer_out = conv(
            layer_out, kernel_size=3, stride=1, filters_out=32, op_name="conv2"
        )

        logits = dense(layer_out, units_out=10, op_name="dense")

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # When using Adam in FP16, you should check
    #     the default value of epsilon and ensure
    #     that it does not underflow
    optimizer = tf.train.AdamOptimizer(args.learning_rate, epsilon=1e-4)

    # Scale loss
    loss *= args.loss_scaling_factor

    # Calculate gradients with scaled loss
    grads_and_vars = optimizer.compute_gradients(loss=loss)

    # Rescale gradients to correct values so parameter update step is correct
    grads_and_vars = [
        (gradient / args.loss_scaling_factor, variable)
        for gradient, variable in grads_and_vars
    ]

    # Apply gradients
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

    # Return loss to original value before reporting it
    loss /= args.loss_scaling_factor

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op
    )


# Configure device with 1 IPU

ipu_configuration = ipu.config.IPUConfig()

ipu_configuration.auto_select_ipus = 1

if args.use_float16_partials:

    ipu_configuration.matmul.poplar_options = {"partialsType": "half"}

    ipu_configuration.convolutions.poplar_options = {"partialsType": "half"}

# Enable all floating-point exceptions
ipu_configuration.floating_point_behaviour.nanoo = True
ipu_configuration.floating_point_behaviour.oflo = True
ipu_configuration.floating_point_behaviour.inv = True
ipu_configuration.floating_point_behaviour.div0 = True


# Use IPU configuration and IPUEstimatorSpec to create IPUEstimator

ipu_run_configuration = ipu.ipu_run_config.IPURunConfig(
    iterations_per_loop=batches_per_epoch, ipu_options=ipu_configuration
)

run_configuration = ipu.ipu_run_config.RunConfig(
    log_step_count_steps=batches_per_epoch, ipu_run_config=ipu_run_configuration
)

ipu_estimator = ipu.ipu_estimator.IPUEstimator(
    config=run_configuration, model_fn=make_estimator_spec
)


# We need an "input function" to train using an estimator
def make_dataset_from_generator():

    # If we use Dataset.from_tensor_slices(), the data will be embedded
    # into the graph as constants. This would make the training graph very
    # large and impractical, so we use Dataset.from_generator() here instead.

    def dataset_generator():
        return zip(x_train, y_train)

    training_data_types = (x_train.dtype, y_train.dtype)
    training_data_shapes = (x_train.shape[1:], y_train.shape[1:])

    dataset = tf.data.Dataset.from_generator(
        dataset_generator, training_data_types, training_data_shapes
    )

    dataset = dataset.prefetch(len(x_train)).cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(len(x_train))
    dataset = dataset.batch(args.batch_size, drop_remainder=True)

    return dataset


# Training progress is logged at the INFO level
tf.logging.set_verbosity(tf.logging.INFO)

ipu_estimator.train(
    input_fn=make_dataset_from_generator, steps=args.epochs * batches_per_epoch
)

print("Program ran successfully")
