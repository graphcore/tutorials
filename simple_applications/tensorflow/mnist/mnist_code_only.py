# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
import time

tf.disable_eager_execution()
tf.disable_v2_behavior()

BATCHSIZE = 32
EPOCHS = 5

mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()

# Cast and normalize the training data
x_train = x_train.astype('float32') / 255
y_train = y_train.astype('int32')

# Build iterator over the data
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().batch(BATCHSIZE, drop_remainder=True)
dataset_iterator = tf.data.make_initializable_iterator(dataset)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])
    return model


model = create_model()


def training_loop_body(x, y):

    logits = model(x, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)

    return([loss, train_op])


# Get inputs from get_next() method of iterator
(x, y) = dataset_iterator.get_next()

with ipu.scopes.ipu_scope('/device:IPU:0'):

    training_loop_body_on_ipu = ipu.ipu_compiler.compile(computation=training_loop_body, inputs=[x, y])

ipu_configuration = ipu.config.IPUConfig()
ipu_configuration.auto_select_ipus = 1
ipu_configuration.configure_ipu_system()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(dataset_iterator.initializer)

    batches_per_epoch = len(x_train) // BATCHSIZE

    for epoch in range(EPOCHS):

        loss_running_total = 0.0

        epoch_start_time = time.time()

        for batch in range(batches_per_epoch):

            # This part runs on IPU since train_loop_body
            # is placed under ipu_scope
            loss = sess.run(training_loop_body_on_ipu)

            loss_running_total += loss[0]

        # Print average loss and time taken for epoch
        print('\n', end='')
        print("Loss:", loss_running_total / batches_per_epoch)
        print("Time:", time.time() - epoch_start_time)

print("Program ran successfully")

# Generated:2022-03-23T10:17 Source:mnist.py SST:0.0.5
