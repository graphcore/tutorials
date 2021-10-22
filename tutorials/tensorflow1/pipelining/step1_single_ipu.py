# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.python import ipu


tf.disable_eager_execution()
tf.disable_v2_behavior()


def parse_args():
    # Handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32,
                        help="The batch size.")
    parser.add_argument("--repeat-count", type=int, default=160,
                        help="The number of times the pipeline will be executed for each step.")
    parser.add_argument("--epochs", type=float, default=50,
                        help="Total number of epochs to train for.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Total number of steps to train for (overrides epochs).")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="The learning rate used with stochastic gradient descent.")
    parser.add_argument("--batches-to-accumulate", type=int, default=16,
                        help="How many batches to process before processing gradients and updating weights.")
    args = parser.parse_args()
    return args


def create_dataset(args):
    # Prepare a TensorFlow dataset with MNIST data
    train_data, _ = mnist.load_data()

    def normalise(x, y):
        return x.astype("float32") / 255.0, y.astype("int32")

    x_train, y_train = normalise(*train_data)

    def generator():
        return zip(x_train, y_train)

    types = (x_train.dtype, y_train.dtype)
    shapes = (x_train.shape[1:], y_train.shape[1:])

    num_examples = len(x_train)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    # Use 'drop_remainder=True' because XLA (and the compiled static IPU graph)
    # expect a complete, fixed sized, set of data as input.
    # Caching and prefetching are important to prevent the host data
    # feed from being the bottleneck for throughput.
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    dataset = dataset.shuffle(num_examples)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return num_examples, dataset


def model(learning_rate, images, labels):
    # Receiving images,labels (x args.batch_size) via infeed.
    # The scoping here helps clarify the execution trace when using --profile.
    with tf.variable_scope("flatten"):
        activations = layers.Flatten()(images)
    with tf.variable_scope("dense256"):
        activations = layers.Dense(256, activation=tf.nn.relu)(activations)
    with tf.variable_scope("dense128"):
        activations = layers.Dense(128, activation=tf.nn.relu)(activations)
    with tf.variable_scope("dense64"):
        activations = layers.Dense(64, activation=tf.nn.relu)(activations)
    with tf.variable_scope("dense32"):
        activations = layers.Dense(32, activation=tf.nn.relu)(activations)
    with tf.variable_scope("logits"):
        logits = layers.Dense(10)(activations)
    with tf.variable_scope("softmax_ce"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
    with tf.variable_scope("mean"):
        loss = tf.reduce_mean(cross_entropy)
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if args.batches_to_accumulate > 1:
            optimizer = ipu.optimizers. \
                GradientAccumulationOptimizerV2(
                    optimizer,
                    num_mini_batches=args.batches_to_accumulate)
        train_op = optimizer.minimize(loss=loss)
    return learning_rate, train_op, outfeed_queue.enqueue(loss)


# Run the training step `args.repeat_count` times by
# iterating the model in an IPU repeat loop.
def loop_repeat_model(learning_rate):
    r = ipu.loops.repeat(args.repeat_count, model, [learning_rate], infeed_queue)
    return r


if __name__ == "__main__":
    args = parse_args()

    num_examples, dataset = create_dataset(args)
    num_train_examples = int(args.epochs * num_examples)

    # Create the data queues from/to IPU
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    # With batch size BS and repeat count RPT,
    # at every step n = (BS * RPT) examples are used.
    # Ensure we process a whole multiple of the batch accumulation count.
    remainder = args.repeat_count % args.batches_to_accumulate
    if remainder > 0:
        args.repeat_count += args.batches_to_accumulate - remainder
        print(f'Rounding up repeat count to whole multiple of '
              f'batches-to-accumulate (== {args.repeat_count})')
    examples_per_step = args.batch_size * args.repeat_count

    # In order to evaluate at least N total examples, do ceil(N / n) steps
    steps = args.steps if args.steps is not None else \
        (num_train_examples + examples_per_step - 1) // examples_per_step
    training_samples = steps * examples_per_step
    print(f'Steps {steps} x examples per step {examples_per_step} '
          f'(== {training_samples} training examples, {training_samples/num_examples} '
          f'epochs of {num_examples} examples)')

    with tf.device('cpu'):
        learning_rate = tf.placeholder(np.float32, [])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        compiled_model = ipu.ipu_compiler.compile(loop_repeat_model, inputs=[learning_rate])

    outfeed_op = outfeed_queue.dequeue()

    ipu.utils.move_variable_initialization_to_cpu()
    init_op = tf.global_variables_initializer()

    # Configure the IPU.
    ipu_configuration = ipu.config.IPUConfig()
    ipu_configuration.auto_select_ipus = 1
    ipu_configuration.configure_ipu_system()

    with tf.Session() as sess:
        # Initialize
        sess.run(init_op)
        sess.run(infeed_queue.initializer)
        # Run
        begin = time.time()
        for step in range(steps):
            sess.run(compiled_model, {learning_rate: args.learning_rate})
            # Read the outfeed for the training losses
            losses = sess.run(outfeed_op)
            if losses is not None and len(losses):
                epoch = float(examples_per_step * step / num_examples)
                if (step == (steps-1) or (step % 10) == 0):
                    print("Step {}, Epoch {:.1f}, Mean loss: {:.3f}".format(
                        step, epoch, np.mean(losses)))
        end = time.time()
        elapsed = end - begin
        samples_per_second = training_samples/elapsed
        print("Elapsed {}, {} samples/sec".format(elapsed, samples_per_second))

    print("Program ran successfully")
