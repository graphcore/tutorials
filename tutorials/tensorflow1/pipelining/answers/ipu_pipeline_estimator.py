# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import os
import time
import math
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.python import ipu
from tensorflow.python.ipu.ipu_session_run_hooks import IPULoggingTensorHook

tf.disable_eager_execution()
tf.disable_v2_behavior()


def parse_args():
    # Handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32,
                        help="The batch size.")
    parser.add_argument("--repeat-count", type=int, default=10,
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
    return x_train, y_train


def layer1_flatten(images, labels):
    with tf.variable_scope("flatten"):
        activations = layers.Flatten()(images)
        return activations, labels


def layer2_dense256(activations, labels):
    with tf.variable_scope("flatten"):
        activations = layers.Dense(256, activation=tf.nn.relu)(activations)
        return activations, labels


def layer3_dense128(activations, labels):
    with tf.variable_scope("dense128"):
        activations = layers.Dense(128, activation=tf.nn.relu)(activations)
        return activations, labels


def layer4_dense64(activations, labels):
    with tf.variable_scope("dense64"):
        activations = layers.Dense(64, activation=tf.nn.relu)(activations)
        return activations, labels


def layer5_dense32(activations, labels):
    with tf.variable_scope("dense32"):
        activations = layers.Dense(32, activation=tf.nn.relu)(activations)
        return activations, labels


def layer6_logits(activations, labels):
    with tf.variable_scope("logits"):
        logits = layers.Dense(10)(activations)
        return logits, labels


def layer7_cel(logits, labels):
    with tf.variable_scope("softmax_ce"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
    with tf.variable_scope("mean"):
        loss = tf.reduce_mean(cross_entropy)
        return loss


def model_fn(mode, params):

    if not mode == tf.estimator.ModeKeys.TRAIN:
        raise NotImplementedError(mode)

    # Defines a pipelined model which is split accross two stages
    def stage1(images, labels):
        r = layer1_flatten(images, labels)
        r = layer2_dense256(*r)
        r = layer3_dense128(*r)
        r = layer4_dense64(*r)
        return r

    def stage2(*r):
        r = layer5_dense32(*r)
        r = layer6_logits(*r)
        loss = layer7_cel(*r)
        return loss

    def optimizer_function(loss):
        # Optimizer function used by the pipeline to automatically set up
        # the gradient accumulation and weight update steps
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
        return ipu.pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

    return ipu.ipu_pipeline_estimator.IPUPipelineEstimatorSpec(
        mode,
        computational_stages=[stage1, stage2],
        optimizer_function=optimizer_function,
        gradient_accumulation_count=params["gradient_accumulation_count"])


if __name__ == "__main__":
    args = parse_args()

    x_train, y_train = create_dataset(args)
    num_examples = len(x_train)

    num_train_examples = int(args.epochs * num_examples)
    batch_size = args.batch_size
    batches = num_train_examples // batch_size

    # With gradient accumulation count GAC and repeat count RPT,
    # IPUPipelineEstimator will iterate (GAC * RPT) steps.
    iterations_per_loop = args.batches_to_accumulate * args.repeat_count

    # IPUPipelineEstimator.train `steps` argument must be a whole multiple of `iterations_per_loop`
    steps = args.steps if args.steps is not None else \
        iterations_per_loop * (batches // iterations_per_loop)
    training_samples = steps * batch_size
    print(f'Steps {steps} x batch size {batch_size} '
          f'(== {training_samples} training examples, {training_samples/num_examples} '
          f'epochs of {num_examples} examples)')

    num_ipus_in_pipeline = 2

    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = num_ipus_in_pipeline
    ipu_config.selection_order = ipu.utils.SelectionOrder.SNAKE

    ipu_run_config = ipu.ipu_run_config.IPURunConfig(
        num_shards = num_ipus_in_pipeline,
        iterations_per_loop = iterations_per_loop,
        ipu_options = ipu_config
    )

    config = ipu.ipu_run_config.RunConfig(
        ipu_run_config = ipu_run_config,
        log_step_count_steps = args.repeat_count
    )

    ipu_estimator = ipu.ipu_pipeline_estimator.IPUPipelineEstimator(
        config = config,
        model_fn = model_fn,
        params = {
            "learning_rate": args.learning_rate,
            "gradient_accumulation_count": args.batches_to_accumulate
        },
    )

    def input_fn():
        types = (x_train.dtype, y_train.dtype)
        shapes = (x_train.shape[1:], y_train.shape[1:])

        def generator():
            return zip(x_train, y_train)

        dataset = tf.data.Dataset.from_generator(generator, types, shapes)
        dataset = dataset.batch(args.batch_size, drop_remainder=True)
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    begin = time.time()
    ipu_estimator.train(input_fn = input_fn, steps = steps)
    end = time.time()
    elapsed = end - begin
    samples_per_second = training_samples/elapsed
    print("Elapsed {}, {} samples/sec".format(elapsed, samples_per_second))

    print("Program ran successfully")
