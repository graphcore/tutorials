# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
from functools import partial
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers

from tensorflow.python import ipu

from outfeed_optimizer import OutfeedOptimizer, OutfeedOptimizerMode
from maybe_outfeed_queue import MaybeOutfeedQueue

tf.disable_v2_behavior()

BATCH_SIZE = 32
LEARNING_RATE = 0.01


def parse_args():
    # Handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-count", type=int, default=100,
                        help="The number of times the model will be executed for each step."
                        "This must be a multiple of the gradient accumulation count."
                        "Set to a small value (such as 1) when profiling.")
    parser.add_argument("--epochs", type=float, default=3,
                        help="Total number of epochs to train for.")
    parser.add_argument('--gradient-accumulation-count', type=int, default=1,
                        help="The number of mini-batches for which gradients will be"
                        " accumulated before a weight update.")
    parser.add_argument('--outfeed-pre-accumulated-gradients', action='store_true',
                        help="Outfeed the pre-accumulated rather than accumulated gradients."
                        " This only makes a difference if the gradient accumulation count is"
                        " greater than one.")
    parser.add_argument('--run-single-step', action="store_true",
                        help="Shorten the run for profiling: runs for a single step.")
    args = parser.parse_args()
    return args


def create_dataset():
    # Prepare a tf dataset with mnist data
    train_data, _ = mnist.load_data()

    def normalise(x, y):
        return x.astype("float32") / 255.0, y.astype("int32")

    x_train, y_train = normalise(*train_data)

    def generator():
        return zip(x_train, y_train)

    types = (x_train.dtype, y_train.dtype)
    shapes = (x_train.shape[1:], y_train.shape[1:])

    n_examples = len(x_train)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.shuffle(n_examples)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return n_examples, dataset


def model(lr, gradient_accumulation_count, outfeed_queue,
          activations_outfeed_queue, optimizer_outfeed_queue,
          outfeed_optimizer_mode, images, labels,):
    with tf.variable_scope("FCModel", use_resource=True):
        x = layers.Flatten()(images)
        x = layers.Dense(256, activation=tf.nn.relu, name="dense1")(x)
        activations_outfeed_queue.maybe_outfeed("dense1", x)
        x = layers.Dense(128, activation=tf.nn.relu, name="dense2")(x)
        activations_outfeed_queue.maybe_outfeed("dense2", x)

        logits = layers.Dense(10, name="dense3")(x)
        activations_outfeed_queue.maybe_outfeed("dense3", logits)
        enqueue = activations_outfeed_queue.maybe_enqueue()
        with tf.control_dependencies([enqueue or tf.no_op()]):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.GradientDescentOptimizer(lr)

        # Wrap the optimizer to outfeed the gradients for selected layers.
        # OutfeedOptimizerMode.BEFORE_APPLY will enqueue the accumulated gradients.
        # OutfeedOptimizerMode.AFTER_COMPUTE will enqueue the individual gradients.
        optimizer = OutfeedOptimizer(optimizer, optimizer_outfeed_queue,
                                     outfeed_optimizer_mode=outfeed_optimizer_mode)

        if gradient_accumulation_count > 1:
            optimizer = ipu.optimizers.GradientAccumulationOptimizerV2(
                    optimizer, gradient_accumulation_count)

        train_op = optimizer.minimize(loss)

    return outfeed_queue.enqueue(loss), train_op


def print_vals(vals, step):
    data = []
    index = 0
    name_length = np.max([len(name) for name in vals.keys()]) + 5
    for val_name, val in vals.items():
        data_item = [index]
        index += 1
        data_item.append(val_name)
        data_item.append(f'{np.mean(val):<4.6f}')  # means
        data_item.append(f'{np.std(val.astype(np.float64)):<4.6f}')  # stds
        data_item.append(f'{np.min(val):<4.6f}')  # min extreme
        data_item.append(f'{np.max(val):<4.6f}')  # max extreme
        data_item.append(f'{np.isnan(val).any()}')  # nans?
        data_item.append(f'{np.isinf(val).any()}')  # infs?
        data.append(data_item)

    print(f"\nStep {step} - Summary Stats")
    print(f'{"Index":<5} {"Name":<{name_length}} {"Mean":<12} {"Std":<12}'
          ' {"Minimum":<12} {"Maximum":<12} {"NaNs":<7} {"infs":<7}')
    for index, name, avg, std, dmin, dmax, nans, infs in data:
        print(f"{index:<5} {name:<{name_length}} {avg:<12} {std:<12} \
              {dmin:<12} {dmax:<12} {nans:<7} {infs:<7}")
    print()

if __name__ == "__main__":
    args = parse_args()

    if args.repeat_count % args.gradient_accumulation_count != 0:
        raise ValueError("--repeat-count (%d) must be a multiple of"
                         " --gradient-accumulation-count (%d)" %
                         (args.repeat_count, args.gradient_accumulation_count))

    print(args)

    if args.outfeed_pre_accumulated_gradients:
        outfeed_optimizer_mode = OutfeedOptimizerMode.AFTER_COMPUTE
    else:
        outfeed_optimizer_mode = OutfeedOptimizerMode.BEFORE_APPLY

    n_examples, dataset = create_dataset()

    # Create the data queues to/from the IPU
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    # Create the outfeed queue for selected gradients
    optimizer_outfeed_queue = MaybeOutfeedQueue(filters=["dense1", "dense2"])

    # Create the outfeed queue for selected activations
    # The filters argument is optional
    activations_outfeed_queue = MaybeOutfeedQueue(filters=["dense1", "dense3"])

    # With batch size BS and repeat count RPT,
    # at every step n = (BS * RPT) examples are used.
    # So in order to evaluate at least N total examples, do ceil(N / n) steps
    num_train_examples = int(args.epochs * n_examples)
    examples_per_step = BATCH_SIZE * args.repeat_count
    steps = ((num_train_examples - 1) // examples_per_step) + 1

    if args.run_single_step:
        steps = 1

    with tf.device('cpu'):
        lr = tf.placeholder(np.float32, [])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        def training_loop():
            return ipu.loops.repeat(args.repeat_count,
                                    partial(model, lr, args.gradient_accumulation_count,
                                            outfeed_queue, activations_outfeed_queue,
                                            optimizer_outfeed_queue, outfeed_optimizer_mode),
                                    infeed_queue=infeed_queue)
        compiled_model = ipu.ipu_compiler.compile(training_loop)

    outfeed_op = outfeed_queue.dequeue()

    # Get the dequeue op (or None) for each MaybeOutfeedQueue object
    # (maybe_dequeue() returns None if nothing was enqueued)
    optimizer_outfeed_op = optimizer_outfeed_queue.maybe_dequeue()
    activations_outfeed_queue_op = activations_outfeed_queue.maybe_dequeue()

    ipu.utils.move_variable_initialization_to_cpu()
    init_op = tf.global_variables_initializer()

    # Configure the IPU device
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    with tf.Session() as sess:
        # Initialize
        sess.run(init_op)
        sess.run(infeed_queue.initializer)

        # Run
        for step in range(steps):
            sess.run(compiled_model, {lr: LEARNING_RATE})
            # Read the outfeed for the training losses
            losses = sess.run(outfeed_op)

            # Read any activations that have been added to the activations outfeed queue
            if activations_outfeed_queue_op:
                activations = sess.run(activations_outfeed_queue_op)

            for k, v in activations.items():
                # The first dimension will be args.gradient_accumulation_count * args.repeat_count
                # The second dimension is BATCH_SIZE
                print(f"Activation key: {k} shape: {v.shape}")

            # Print statistics for the selected activations
            # cast to float32 to avoid overflow when calculating statistics
            activations = {k: v.astype(np.float32) for k, v in activations.items()}
            print_vals(activations, step)

            # Read any gradients that have been added to the optimizer outfeed queue
            if optimizer_outfeed_op:
                gradients = sess.run(optimizer_outfeed_op)
                for k, v in gradients.items():
                    # If using OutfeedOptimizerMode.BEFORE_APPLY then the first dimension will be args.repeat_count
                    # If using OutfeedOptimizerMode.AFTER_COMPUTE it will be args.gradient_accumulation_count * args.repeat_count
                    print(f"Gradient key: {k} shape: {v.shape}")

                # Print statistics for the selected gradients
                # cast to float32 to avoid overflow when calculating statistics
                gradients = {k: v.astype(np.float32) for k, v in gradients.items()}
                print_vals(gradients, step)

            epoch = float(examples_per_step * step / n_examples)
            print("Epoch {:.1f}, Mean loss: {:.3f}\n".format(
                epoch, np.mean(losses)))
