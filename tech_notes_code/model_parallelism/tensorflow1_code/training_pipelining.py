# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.ipu import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.keras import layers
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# default data_format is 'channels_last'
dataset = Dataset.from_tensor_slices(
    (tf.random.uniform([2, 128, 128, 3], dtype=tf.float32),
     tf.random.uniform([2], maxval=10, dtype=tf.int32))
    )
dataset = dataset.batch(batch_size=2, drop_remainder=True)
dataset = dataset.shuffle(1000)
dataset = dataset.cache()
dataset = dataset.repeat()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Create the data queues from/to IPU.
infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()


# Create a pipelined model which is split accross four stages.
def stage1(x, labels):
    with variable_scope.variable_scope("stage1", use_resource=True):
        with variable_scope.variable_scope("conv", use_resource=True):
            x = layers.Conv2D(3, 1)(x)
            return x, labels


def stage2(x, labels):
    with variable_scope.variable_scope("stage2", use_resource=True):
        with variable_scope.variable_scope("conv", use_resource=True):
            x = layers.Conv2D(3, 1)(x)
            return x, labels


def stage3(x, labels):
    with variable_scope.variable_scope("stage3", use_resource=True):
        with variable_scope.variable_scope("conv", use_resource=True):
            x = layers.Conv2D(3, 1)(x)
            return x, labels


def stage4(x, labels):
    with variable_scope.variable_scope("stage3", use_resource=True):
        with variable_scope.variable_scope("flatten", use_resource=True):
            x = layers.Flatten()(x)
        with variable_scope.variable_scope("dense", use_resource=True):
            logits = layers.Dense(10)(x)
        with variable_scope.variable_scope("entropy", use_resource=True):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        with variable_scope.variable_scope("loss", use_resource=True):
            loss = tf.reduce_mean(cross_entropy)
        return loss


def optimizer_function(loss):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)


def my_net():
    pipeline_op = pipelining_ops.pipeline(
                        computational_stages=[stage1, stage2, stage3, stage4],
                        gradient_accumulation_count=8,
                        repeat_count=2,
                        inputs=[],
                        infeed_queue=infeed_queue,
                        outfeed_queue=outfeed_queue,
                        optimizer_function=optimizer_function,
                        name="Pipeline")
    return pipeline_op


with ops.device("/device:IPU:0"):
    r = ipu_compiler.compile(my_net, inputs=[])

dequeue_op = outfeed_queue.dequeue()

cfg = config.IPUConfig()
cfg.allow_recompute = True
cfg.selection_order = config.SelectionOrder.ZIGZAG
cfg.auto_select_ipus = 4
cfg.configure_ipu_system()
utils.move_variable_initialization_to_cpu()

with tf.Session() as sess:
    sess.run(variables.global_variables_initializer())
    sess.run(infeed_queue.initializer)
    sess.run(r)
    losses = sess.run(dequeue_op)
