# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import os
import numpy as np
import time
import logging

# Import IPU API
from tensorflow.python import ipu
from tensorflow_core.python.keras.backend import unique_object_name
from typing import Any, Optional, Tuple, Union


def create_input_data(batch_size=1, height=224, width=224, channels=4):
    """
        Create the input dataset for in-feeds

        :param batch_size: size of the batches to process
        :param height: height of input image
        :param width: width of input image
        :param channels: channels (RGB) in the input image
        :return: Constructed dataset
    """
    # Synthetic input data follows NHWC format
    input_data = np.random.random((batch_size, height, width, channels))
    input_data = tf.cast(input_data, DTYPE)

    # Prepare dataset for ipu_infeeds
    ds = tf.data.Dataset \
        .range(1) \
        .map(lambda k: {"features": input_data}) \
        .repeat() \
        .prefetch(BATCHES_PER_STEP)
    return ds


def conv(input_tensor: tf.Tensor,
         kernel_size: Union[int, Tuple[int, int]],
         filters_out: int,
         stride: Optional[int] = 1,
         padding: Optional[str] = 'SAME',
         add_bias: Optional[bool] = True,
         dtype: Optional[Any] = tf.float16,
         name: Optional[str] = None,
         weight_suffix: Optional[str] = "kernel",
         bias_suffix: Optional[str] = "conv/bias",
         *_):
    """
        Apply convolutional layer and optional bias on input tensor.

        Args:
            input_tensor: Input data
            kernel_size: Filter size (assumes equal height and width)
            filters_out: Number of output filters
            stride: Stride of the filter
            padding: Type of padding to use
            add_bias: Should bias be added
            dtype: Data type of parameters
            name: Optional name for this op
            weight_suffix: String to weight name with
            bias_suffix: String to suffix the bias name with

        Returns: Output of convolution operator.
    """

    # Assumes input in NHWC format.
    filters_in = input_tensor.get_shape()[-1]
    if isinstance(kernel_size, int):
        w_shape = [kernel_size, kernel_size, filters_in, filters_out]
    else:
        w_shape = kernel_size + (filters_in, filters_out)
    w_init = tf.contrib.layers.xavier_initializer(dtype=dtype)
    if name is None:
        name = unique_object_name("conv2d", zero_based=True)

    name_scope = tf.get_default_graph().get_name_scope()
    if name_scope not in ["", None]:
        name = name_scope + "/" + name

    with tf.get_default_graph().as_default():
        with tf.variable_scope(name):
            weights = tf.get_variable(weight_suffix,
                                      shape=w_shape,
                                      initializer=w_init,
                                      dtype=dtype)

    output_tensor = tf.nn.conv2d(input_tensor,
                                 weights, [1, stride, stride, 1],
                                 padding=padding.upper(),
                                 name=name)

    if add_bias:
        b_shape = [filters_out]
        b_init = tf.zeros_initializer()
        with tf.variable_scope(name):
            biases = tf.get_variable(bias_suffix,
                                     shape=b_shape,
                                     initializer=b_init,
                                     dtype=dtype)
        output_tensor += biases
    return output_tensor


# Block definitions for ResNeXt
def input_block(x):
    x = conv(x, kernel_size=7, stride=2, filters_out=64, name="conv1")
    x = norm(x, training=False)
    x = relu(x)
    x = maxpool(x, size=3, stride=2)
    return x


def maxpool(x, size=3, stride=2):
    x = tf.nn.max_pool(x,
                       ksize=[1, size, size, 1],
                       strides=[1, stride, stride, 1],
                       padding='SAME')
    return x


def reduce_mean(x, indices=(1, 2)):
    x = tf.reduce_mean(x, reduction_indices=indices)
    return x


def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    w_init = tf.contrib.layers.xavier_initializer(dtype=tf.float16)
    b_init = tf.constant_initializer(0.0)

    weights = tf.get_variable('weights',
                              shape=[num_units_in, num_units_out],
                              initializer=w_init,
                              dtype=tf.float16)
    biases = tf.get_variable('biases',
                             shape=[num_units_out],
                             initializer=b_init,
                             dtype=tf.float16)

    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def norm(x, training=False):
    x = tf.layers.batch_normalization(x,
                                      fused=True,
                                      center=True,
                                      scale=True,
                                      training=training,
                                      trainable=training,
                                      momentum=0.997,
                                      epsilon=1e-5)
    return x


def relu(x):
    return tf.nn.relu(x)


def group_conv(x,
               ksize,
               stride,
               filters_in,
               filters_out,
               index=0,
               groups=1,
               dtype=tf.float16,
               name='conv'):
    """
    Apply group convolutions by leveraging XLA implementation.

    """
    with tf.variable_scope(name, use_resource=True):
        W = tf.get_variable(
            "conv2d/kernel" + str(index),
            shape=[ksize, ksize, filters_in.value / groups, filters_out],
            dtype=dtype,
            trainable=True,
            initializer=tf.variance_scaling_initializer())
        return tf.nn.conv2d(x,
                            filters=W,
                            strides=[1, stride, stride, 1],
                            padding='SAME')


def group_conv_block(x, first_stride, filters, count, name='', cardinality=4):
    """
        Group convolution block implementation.

        :param x: Input tensor
        :param first_stride: Initial tensor
        :param filters: List of number of filters for various convolution blocks
        :param count: Number of times block is repeated
        :param name: Name of block
        :param cardinality: Number of groups = outputchannels/cardinality
        :return: Layer x after application of all the ops within the block
    """
    for i in range(count):
        shortcut = x
        stride = (first_stride if (i == 0) else 1)

        # First vanilla convolution
        x = conv(x,
                 kernel_size=1,
                 stride=stride,
                 filters_out=filters[0],
                 add_bias=False,
                 name=name + str(i) + "_1",
                 dtype=tf.float16)
        x = norm(x)
        x = relu(x)

        # Group convolution evaluation
        x = group_conv(x,
                       ksize=3,
                       stride=1,
                       filters_in=x.get_shape()[-1],
                       filters_out=filters[0],
                       index=1,
                       name=name + str(i) + "_2",
                       groups=cardinality,
                       dtype=tf.float16)
        x = norm(x)
        x = relu(x)

        # Second vanilla convolution
        x = conv(x,
                 kernel_size=1,
                 stride=1,
                 filters_out=filters[1],
                 add_bias=False,
                 name=name + str(i) + "_3",
                 dtype=tf.float16)
        x = norm(x)
        if i == 0:
            shortcut = conv(shortcut,
                            kernel_size=1,
                            stride=stride,
                            filters_out=filters[1],
                            add_bias=False,
                            name=name + str(i) + "skip",
                            dtype=tf.float16)
            shortcut = norm(shortcut)
        x = shortcut + x
        x = relu(x)
    return x


def resnext101_model():
    """
    Define ResNext-101 network graph

    """
    def body(features):
        with tf.variable_scope("VanillaResNeXt"):
            x = input_block(features)
            x = group_conv_block(x,
                                 first_stride=1,
                                 filters=[128, 256],
                                 count=3,
                                 cardinality=CARDINALITY,
                                 name='res2_')  # 112
            x = group_conv_block(x,
                                 first_stride=2,
                                 filters=[256, 512],
                                 count=4,
                                 cardinality=CARDINALITY,
                                 name='res3_')  # 224
            x = group_conv_block(x,
                                 first_stride=2,
                                 filters=[512, 1024],
                                 count=23,
                                 cardinality=CARDINALITY,
                                 name='res4_')  # 448
            x = group_conv_block(x,
                                 first_stride=2,
                                 filters=[1024, 2048],
                                 count=3,
                                 cardinality=CARDINALITY,
                                 name='res5_')  # 896
            x = reduce_mean(x)
            output = fc(x, num_units_out=1000)
            outfeed = outfeed_queue.enqueue(output)
            return outfeed

    return tf.python.ipu.loops.repeat(n=BATCHES_PER_STEP,
                                      body=body,
                                      infeed_queue=infeed_queue)


if __name__ == '__main__':
    print("ResNeXt-101 Inference")

    IPU_MODEL = False

    # Number of steps
    NUM_ITERATIONS = 5
    BATCHES_PER_STEP = 1000

    # Model
    MODEL = 'ResNeXt-101'
    CARDINALITY = 32
    BATCH_SIZE = 4

    # Precision
    DTYPE = tf.float16

    # Create input data using randomized numpy arrays
    dataset = create_input_data(batch_size=BATCH_SIZE,
                                height=224,
                                width=224,
                                channels=4)

    if IPU_MODEL:
        os.environ['TF_POPLAR_FLAGS'] = "--use_ipu_model"

    # Setup infeed queue
    if BATCHES_PER_STEP > 1:
        with tf.device('cpu'):
            infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(
                dataset)
    else:
        raise NotImplementedError("batches per step == 1 not implemented yet.")

    # Setup outfeed
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    # Compiles graph and targets IPU(s)
    with ipu.scopes.ipu_scope('/device:IPU:0'):
        res = ipu.ipu_compiler.compile(resnext101_model, inputs=[])

    # Setup IPU configuration and build session
    cfg = ipu.config.IPUConfig()
    cfg.convolutions.poplar_options["availableMemoryProportion"] = "0.3"
    cfg.ipu_model.compile_ipu_code = False
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()
    ipu.utils.move_variable_initialization_to_cpu()
    outfeed = outfeed_queue.dequeue()

    with tf.Session() as sess:
        fps = []
        latency = []
        sess.run(infeed_queue.initializer)
        sess.run(tf.global_variables_initializer())
        # Warm up
        print("Compiling and Warmup...")
        start = time.time()
        sess.run(res)
        outfed = sess.run(outfeed)
        duration = time.time() - start
        print("Duration: {:.3f} seconds\n".format(duration))
        for iter_count in range(NUM_ITERATIONS):
            print("Running iteration: ", iter_count)
            # Run
            start = time.time()
            sess.run(res)
            sess.run(outfeed)
            stop = time.time()
            fps.append((BATCHES_PER_STEP * BATCH_SIZE) / (stop - start))
            logging.info(
                "Iter {3}: {0} Throughput using real data = {1:.1f}"
                " imgs/sec at batch size = {2}".format(
                    str(MODEL), fps[-1], BATCH_SIZE, iter_count))
            latency.append(1000 * (stop - start) / BATCHES_PER_STEP)
            logging.info(
                "Iter {3}: {0} Latency using real data = {2:.2f} msecs "
                "at batch_size = {1}".format(str(MODEL), BATCH_SIZE,
                                             latency[-1], iter_count))

        print("Average statistics over {0} iterations, excluding the 1st "
              "iteration.".format(NUM_ITERATIONS))
        print("-------------------------")
        fps = fps[1:]
        latency = latency[1:]
        print(
            "Throughput at bs={} of {}: min={}, max={}, mean={}, std={}.".
            format(BATCH_SIZE, str(MODEL), min(fps), max(fps),
                   np.mean(fps), np.std(fps)))
        print("Latency at bs={} of {}: min={}, max={}, mean={}, std={}.".
              format(BATCH_SIZE, str(MODEL), min(latency), max(latency),
                     np.mean(latency), np.std(latency)))
