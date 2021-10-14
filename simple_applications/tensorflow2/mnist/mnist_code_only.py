# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow import keras

from tensorflow.python import ipu

if tf.__version__[0] != '2':
    raise ImportError("TensorFlow 2 is required for this example")

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

plt.figure(figsize=(15, 10))
for i, image, label in zip(range(15), x_train, y_train):
    ax = plt.subplot(5, 5, i + 1)
    ax.set_title(label)
    plt.imshow(image)
plt.tight_layout()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000).batch(32, drop_remainder=True)
train_ds = train_ds.map(
    lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32))
)
train_ds = train_ds.repeat()

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')])
    return model

cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()

strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
    model = create_model()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(),
        steps_per_execution=100
    )
    model.fit(train_ds, steps_per_epoch=2000, epochs=4)
