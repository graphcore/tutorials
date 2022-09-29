# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from tensorflow.python.keras.layers.pooling import MaxPool2D
import popdist
import popdist.tensorflow

import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod import popdist_strategy

tf.disable_v2_behavior()

BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 100


def initialize_model():
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10),
        ]
    )


# Initialize IPU configuration.
config = ipu.config.IPUConfig()
popdist.tensorflow.set_ipu_config(config)
config.configure_ipu_system()

hvd.init()

# Create distribution strategy.
strategy = popdist_strategy.PopDistStrategy()

# Get and normalize the training data.
(train_x, train_y), _ = tf.keras.datasets.cifar10.load_data()
train_x = train_x.astype(np.float32) / 255.0
train_y = train_y.astype(np.int32)

# Calculate global batch size and number of iterations.
num_total_replicas = popdist.getNumTotalReplicas()
global_batch_size = num_total_replicas * BATCH_SIZE
num_iterations = (
    len(train_y) // global_batch_size // num_total_replicas
) * num_total_replicas

# Create dataset and shard it across the instances.
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.shard(
    num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex()
)

# Perform shuffling and batching after sharding.
dataset = dataset.shuffle(buffer_size=len(train_y) // popdist.getNumInstances())
dataset = dataset.repeat()
dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)

with strategy.scope():
    model = initialize_model()

    # The TensorFlow optimizers are distribution-aware, meaning that they will
    # automatically insert cross-replica sums of the gradients when invoked inside
    # the scope of a distribution strategy. Since we are also inside an IPU scope,
    # this cross-replica sum will be performed on the IPU using the Graphcore
    # Communication Library (GCL).
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)

    def per_replica_step(loss_sum, x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true=y, y_pred=logits, from_logits=True
            )

            # Normalize the loss by the global batch size.
            loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=global_batch_size
            )

        loss_sum += loss
        gradients = tape.gradient(loss, model.trainable_variables)
        train_op = optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss_sum, train_op

    def per_replica_loop():
        return ipu.loops.repeat(
            num_iterations, per_replica_step, infeed_queue=infeed_queue, inputs=[0.0]
        )

    def run_model():
        per_replica_loss = strategy.experimental_run_v2(per_replica_loop)

        # Since the loss is already normalized by the global batch size above, we
        # compute the sum rather than the average here. Note that since we are still
        # inside an IPU scope, this will also be performed on the IPU using GCL.
        # Divide by number of iterations to return the average (per-replica) loss across the epoch.
        return (
            strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss)
            / num_iterations
        )

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        compiled_model = ipu.ipu_compiler.compile(run_model)

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
