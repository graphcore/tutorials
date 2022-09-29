# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope

tf.disable_v2_behavior()

NUM_IPUS = 4

# Configure the IPU system
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = NUM_IPUS
cfg.configure_ipu_system()

# Create the CPU section of the graph
with tf.device("cpu"):
    pa = tf.placeholder(np.float32, [2], name="a")
    pb = tf.placeholder(np.float32, [2], name="b")
    pc = tf.placeholder(np.float32, [2], name="c")


# Distribute the computation across four shards
def sharded_graph(pa, pb, pc):
    with ipu.scopes.ipu_shard(0):
        o1 = pa + pb
    with ipu.scopes.ipu_shard(1):
        o2 = pa + pc
    with ipu.scopes.ipu_shard(2):
        o3 = pb + pc
    with ipu.scopes.ipu_shard(3):
        out = o1 + o2 + o3
        return out


# Create the IPU section of the graph
with ipu_scope("/device:IPU:0"):
    result = ipu.ipu_compiler.compile(sharded_graph, [pa, pb, pc])

with tf.Session() as sess:
    # sharded run
    result = sess.run(
        result, feed_dict={pa: [1.0, 1.0], pb: [0.0, 1.0], pc: [1.0, 5.0]}
    )
    print(result)
