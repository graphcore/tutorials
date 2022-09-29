# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.keras import layers
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# default data_format is 'channels_last'
dataset = Dataset.from_tensor_slices(
    np.random.uniform(size=(2, 128, 128, 3)).astype(np.float32)
)
dataset = dataset.batch(batch_size=2, drop_remainder=True)
dataset = dataset.cache()
dataset = dataset.repeat()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Create the data queues from/to IPU.
infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()


# Create a pipelined model which is split across two stages.
def stage1(x):
    x = layers.Conv2D(128, 1)(x)
    return x


def stage2(x):
    x = layers.Conv2D(128, 1)(x)
    return x


def my_net():
    pipeline_op = pipelining_ops.pipeline(
        computational_stages=[stage1, stage2],
        gradient_accumulation_count=16,
        repeat_count=2,
        inputs=[],
        infeed_queue=infeed_queue,
        outfeed_queue=outfeed_queue,
        name="Pipeline",
    )
    return pipeline_op


with ops.device("/device:IPU:0"):
    r = ipu_compiler.compile(my_net, inputs=[])

dequeue_op = outfeed_queue.dequeue()

cfg = config.IPUConfig()
cfg.auto_select_ipus = 2
cfg.configure_ipu_system()
utils.move_variable_initialization_to_cpu()

with tf.Session() as sess:
    sess.run(variables.global_variables_initializer())
    sess.run(infeed_queue.initializer)
    sess.run(r)
    output = sess.run(dequeue_op)
