# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popdist
import popdist.tensorflow
import tensorflow.compat.v1 as tf
import numpy as np

from tensorflow.python import ipu
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod import ipu_multi_replica_strategy
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense,
                                     Activation, Flatten)
from tensorflow.python.ipu.keras.layers import Dropout

tf.disable_v2_behavior()


BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 100

# Configure the IPU system with the PopDist configuration.
config = ipu.config.IPUConfig()
popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
config.configure_ipu_system()

# Initialize Horovod.
hvd.init()

# Create distribution strategy.
strategy = ipu_multi_replica_strategy.IPUMultiReplicaStrategy()

# Get and normalize the training data.
(train_x, train_y), _ = tf.keras.datasets.cifar10.load_data()
train_x = train_x.astype(np.float32) / 255.0
train_y = train_y.astype(np.int32)

# Create dataset and shard it across the instances.
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.shard(
    num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex())

# Perform shuffling and batching after sharding.
dataset = dataset.shuffle(
    buffer_size=len(train_y) // popdist.getNumInstances())
dataset = dataset.repeat()
dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)

global_batch_size = BATCH_SIZE * popdist.getNumTotalReplicas()
steps_per_epoch = len(train_y) // global_batch_size

print(f"Global batch size {global_batch_size}:",
      f"{steps_per_epoch} steps per epoch")

# Create the model under the strategy scope.
with strategy.scope():

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def per_replica_step(loss_sum, x, y):
        # Build a simple convolutional model with the Keras API.
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3))(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x, training=True)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x, training=True)

        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x, training=True)
        logits = Dense(NUM_CLASSES)(x)

        per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=y, y_pred=logits, from_logits=True)

        # Normalize the loss by the global batch size.
        loss = tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=global_batch_size)
        loss_sum += loss

        # The TensorFlow optimizers are distribution-aware, meaning that they will
        # automatically insert cross-replica sums of the gradients when invoked inside
        # the scope of a distribution strategy. Since we are also inside an IPU scope,
        # this cross-replica sum will be performed on the IPU using the Graphcore
        # Communication Library (GCL).
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss)

        return loss_sum, train_op

    def per_replica_loop():
        loss_sum = ipu.loops.repeat(
            steps_per_epoch,
            per_replica_step,
            infeed_queue=infeed_queue,
            inputs=[0.0])
        # Return the average (per-replica) loss across the epoch.
        return loss_sum / steps_per_epoch

    def model():
        per_replica_loss = strategy.experimental_run_v2(per_replica_loop)
        # Since the loss is already normalized by the global batch size above, we
        # compute the sum rather than the average here. Note that since we are still
        # inside an IPU scope, this will also be performed on the IPU using GCL.
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss)

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        compiled_model = ipu.ipu_compiler.compile(model)

    with tf.Session() as sess:
        sess.run(infeed_queue.initializer)
        sess.run(tf.global_variables_initializer())

        for epoch in range(NUM_EPOCHS):
            loss = sess.run(compiled_model)[0]
            print(f"Epoch {epoch + 1}: loss={loss}")

        # Save the model. Each instance should have identical weights, so it should
        # be sufficient to only save from one of the instances. However, saving from
        # all of them can be useful to verify that they are in sync.
        saver = tf.train.Saver()
        saver.save(sess, f"instance_{popdist.getInstanceIndex()}")
