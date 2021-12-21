# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pathlib
import popdist
import popdist.tensorflow

import numpy as np

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod import popdist_strategy

BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 100


def initialize_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, 3, activation='relu'),
        tf.keras.layers.Conv2D(
            32, 3, activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(
            64, 3, activation='relu'),
        tf.keras.layers.Conv2D(
            64, 3, activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ])

# Initialize IPU configuration.
config = ipu.config.IPUConfig()
popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
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
num_iterations = (len(train_y) // global_batch_size // num_total_replicas) * num_total_replicas

# Create dataset and shard it across the instances.
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.shard(
    num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex())

# Perform shuffling and batching after sharding.
dataset = dataset.shuffle(
    buffer_size=len(train_y) // popdist.getNumInstances())
dataset = dataset.repeat()
dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)

with strategy.scope():
    model = initialize_model()
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn,
                  steps_per_execution=num_iterations * num_total_replicas)
    history = model.fit(
        dataset, steps_per_epoch=num_iterations * num_total_replicas, epochs=NUM_EPOCHS)

    saved = model.save(f"instance_{popdist.getInstanceIndex()}.h5")
