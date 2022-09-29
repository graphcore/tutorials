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
    parser.add_argument("--batch-size", type=int, default=32, help="The batch size.")
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=10,
        help="The number of times the pipeline will be executed for each step.",
    )
    parser.add_argument(
        "--epochs", type=float, default=50, help="Total number of epochs to train for."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total number of steps to train for (overrides epochs).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="The learning rate used with stochastic gradient descent.",
    )
    parser.add_argument(
        "--batches-to-accumulate",
        type=int,
        default=16,
        help="How many batches to process before processing gradients and updating weights.",
    )
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


def layer1_flatten(learning_rate, images, labels):
    with tf.variable_scope("flatten"):
        activations = layers.Flatten()(images)
        return learning_rate, activations, labels


def layer2_dense256(learning_rate, activations, labels):
    with tf.variable_scope("flatten"):
        activations = layers.Dense(256, activation=tf.nn.relu)(activations)
        return learning_rate, activations, labels


def layer3_dense128(learning_rate, activations, labels):
    with tf.variable_scope("dense128"):
        activations = layers.Dense(128, activation=tf.nn.relu)(activations)
        return learning_rate, activations, labels


def layer4_dense64(learning_rate, activations, labels):
    with tf.variable_scope("dense64"):
        activations = layers.Dense(64, activation=tf.nn.relu)(activations)
        return learning_rate, activations, labels


def layer5_dense32(learning_rate, activations, labels):
    with tf.variable_scope("dense32"):
        activations = layers.Dense(32, activation=tf.nn.relu)(activations)
        return learning_rate, activations, labels


def layer6_logits(learning_rate, activations, labels):
    with tf.variable_scope("logits"):
        logits = layers.Dense(10)(activations)
        return learning_rate, logits, labels


def layer7_cel(learning_rate, logits, labels):
    with tf.variable_scope("softmax_ce"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
    with tf.variable_scope("mean"):
        loss = tf.reduce_mean(cross_entropy)
        return learning_rate, loss


def optimizer_function(learning_rate, loss):
    # Optimizer function used by the pipeline to automatically set up
    # the gradient accumulation and weight update steps
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    return ipu.pipelining_ops.OptimizerFunctionOutput(optimizer, loss)


def pipelined_model(learning_rate):
    # Defines a pipelined model which is split across two stages
    def stage1(learning_rate, images, labels):
        r = layer1_flatten(learning_rate, images, labels)
        r = layer2_dense256(*r)
        r = layer3_dense128(*r)
        r = layer4_dense64(*r)
        return r

    def stage2(*r):
        r = layer5_dense32(*r)
        r = layer6_logits(*r)
        r = layer7_cel(*r)
        return r

    pipeline_op = ipu.pipelining_ops.pipeline(
        computational_stages=[stage1, stage2],
        gradient_accumulation_count=args.batches_to_accumulate,
        repeat_count=args.repeat_count,
        inputs=[learning_rate],
        infeed_queue=infeed_queue,
        outfeed_queue=outfeed_queue,
        optimizer_function=optimizer_function,
        pipeline_schedule=ipu.pipelining_ops.PipelineSchedule.Grouped,
        outfeed_loss=True,
        name="Pipeline",
    )
    return pipeline_op


if __name__ == "__main__":
    args = parse_args()

    num_examples, dataset = create_dataset(args)
    num_train_examples = int(args.epochs * num_examples)

    # Create the data queues from/to IPU
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    # With batch size BS, gradient accumulation count GAC and repeat count RPT,
    # at every step n = (BS * GAC * RPT) examples are used.
    examples_per_step = args.batch_size * args.batches_to_accumulate * args.repeat_count

    # In order to evaluate at least N total examples, do ceil(N / n) steps
    steps = (
        args.steps
        if args.steps is not None
        else (num_train_examples + examples_per_step - 1) // examples_per_step
    )
    training_samples = steps * examples_per_step
    print(
        f"Steps {steps} x examples per step {examples_per_step} "
        f"(== {training_samples} training examples, {training_samples/num_examples} "
        f"epochs of {num_examples} examples)"
    )

    with tf.device("cpu"):
        learning_rate = tf.placeholder(np.float32, [])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        compiled_model = ipu.ipu_compiler.compile(
            pipelined_model, inputs=[learning_rate]
        )

    outfeed_op = outfeed_queue.dequeue()

    ipu.utils.move_variable_initialization_to_cpu()
    init_op = tf.global_variables_initializer()

    # Configure the IPU.
    ipu_configuration = ipu.config.IPUConfig()
    ipu_configuration.auto_select_ipus = 2
    ipu_configuration.selection_order = ipu.utils.SelectionOrder.SNAKE
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
                if step == (steps - 1) or (step % 10) == 0:
                    print(
                        f"Step {step}, Epoch {epoch:.1f}, Mean loss:"
                        f" {np.mean(losses):.3f}"
                    )
        end = time.time()
        elapsed = end - begin
        samples_per_second = training_samples / elapsed
        print(f"Elapsed {elapsed}, {samples_per_second} samples/sec")

    print("Program ran successfully")
