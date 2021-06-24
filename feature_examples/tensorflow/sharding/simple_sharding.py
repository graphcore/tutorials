# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import argparse

import numpy as np
import tensorflow as tf

from tensorflow.python.ipu import ipu_compiler, scopes, config

NUM_SHARDS = 2

# With sharding all placeholders MUST be explicitly placed on
# the CPU device:
with tf.device("cpu"):
    pa = tf.placeholder(np.float32, [2], name="a")
    pb = tf.placeholder(np.float32, [2], name="b")
    pc = tf.placeholder(np.float32, [2], name="c")


# Put part of the computation on shard 1 and part on shard 2.
# Sharding is automatically enabled on detection of nodes
# placed with 'scopes.ipu_shard(...)':
def manual_sharding(pa, pb, pc):
    with scopes.ipu_shard(0):
        o1 = pa + pb
    with scopes.ipu_shard(1):
        o2 = pa + pc
        out = o1 + o2
        return out


def my_graph(pa, pb, pc):
    result = manual_sharding(pa, pb, pc)
    return result

# Create the IPU section of the graph
with scopes.ipu_scope("/device:IPU:0"):
    out = ipu_compiler.compile(my_graph, [pa, pb, pc])

# Define the feed_dict input data
fd = {pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]}
# Configure an IPU device that has NUM_SHARDS devices that we will
# shard across.
cfg = config.IPUConfig()
cfg.auto_select_ipus = NUM_SHARDS
cfg.configure_ipu_system()

with tf.Session() as sess:
    result = sess.run(out, fd)
    print(result)
