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
    parser.add_argument("--splits", nargs="+", help="Specify splits. " +
                        "Each split specifies the layer that will be assigned to the *next* stage. " +
                        "The layers are: " +
                        ", ".join([layer["id"] for layer in model_layers])+". ",
                        default=["dense32"])
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
            labels=labels, logits=logits)
    with tf.variable_scope("mean"):
        loss = tf.reduce_mean(cross_entropy)
        return learning_rate, loss


def optimizer_function(learning_rate, loss):
    # Optimizer function used by the pipeline to automatically set up
    # the gradient accumulation and weight update steps
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    return ipu.pipelining_ops.OptimizerFunctionOutput(optimizer, loss)


def discover_layers():
    # This parses global functions with form "layer<n>_<id>"
    # e.g. "layer3_BlockA"
    # The layers will be sorted by <n>
    # A layer list of dictionaries is returned with:
    #   "func" : Function reference
    #   "id"   : String name of layer (==<id>)
    layers = []
    global_symbols = globals()
    prefix = "layer"
    layer_funcs = [key for key in global_symbols if key.startswith(prefix)]
    for layer_func in layer_funcs:
        try:
            idx_id = layer_func[len(prefix):]
            idx, id = idx_id.split("_")
            layers.append({"func": global_symbols[layer_func],
                           "id": id, "idx": int(idx)})
        except:
            pass

    def use_idx(e):
        return e["idx"]

    layers.sort(key=use_idx)
    return layers


def move_layers_to_stages(layers, splits):
    # Sequence layers into distinct groups (stages)
    # according to splits.
    def next_split_layer():
        # Returns idx of layer matching id.
        if stage >= len(splits):
            return None
        split = splits[stage]
        try:
            return next(layer for layer in layers if (layer["id"] == split))
        except:
            print("Failed to match split layer with id \"{}\"".format(split))
            return None

    stages = [[]]
    stage = 0
    idx = 0
    next_split = next_split_layer()
    while len(layers):
        ly = layers.pop(0)
        if ly == next_split:
            stage += 1
            stages.append([])
            next_split = next_split_layer()
        stages[stage].append(ly)
    return stages


def pipelined_model(learning_rate):
    # Helper that defines a single stage consisting of one or more layers
    def make_pipeline_stage(idx, stage):
        def _stage(*args):
            for layer in stage:
                with tf.variable_scope("stage"+str(idx)+"_"+layer["id"], use_resource=True):
                    print("Issuing stage {} layer {}".format(idx, layer["id"]))
                    args = layer["func"](*args)
            return args
        return _stage

    # Make each stage (function) and add it to the computational stages
    computational_stages = []
    for idx, stage in enumerate(stages):
        f = make_pipeline_stage(idx, stage)
        computational_stages.append(f)

    pipeline_op = ipu.pipelining_ops.pipeline(
        computational_stages = computational_stages,
        gradient_accumulation_count = args.batches_to_accumulate,
        repeat_count = args.repeat_count,
        inputs = [learning_rate],
        infeed_queue = infeed_queue,
        outfeed_queue = outfeed_queue,
        optimizer_function = optimizer_function,
        pipeline_schedule = ipu.pipelining_ops.PipelineSchedule.Grouped,
        outfeed_loss=True,
        name = "Pipeline")
    return pipeline_op


if __name__ == "__main__":
    model_layers = discover_layers()

    # Show final list of layers
    print("Layers:")
    layer_list = [layer["id"] for layer in model_layers]
    print(" "+(", ".join(layer_list)))

    args = parse_args()

    # Sequence layers into stage-groups
    stages = move_layers_to_stages(model_layers, args.splits)
    if (len(stages) != len(args.splits)+1):
        print("Unexpected stage count - check splits are valid")
        exit(-1)

    # Show final list of staged layers
    print("Stages:")
    for idx, stage in enumerate(stages):
        layer_list = [layer["id"] for layer in stage]
        print(" "+str(idx)+". "+("-".join(layer_list)))
        if len(layer_list) == 0:
            print("Unexpected empty stage - check splits are valid")
            exit(-1)

    # Check stage count is power2.
    num_stages = len(args.splits)+1
    num_ipus = int(math.pow(2, math.ceil(math.log(num_stages, 2))))
    if num_stages != num_ipus:
        print("Stage count must be power2 (specified {} versus next power2 {})".format(
            num_stages, num_ipus))
        exit(-1)

    num_examples, dataset = create_dataset(args)
    num_train_examples = int(args.epochs * num_examples)

    # Create the data queues from/to IPU
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    # With batch size BS, gradient accumulation count GAC and repeat count RPT,
    # at every step n = (BS * GAC * RPT) examples are used.
    examples_per_step = args.batch_size * args.batches_to_accumulate * args.repeat_count

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
        compiled_model = ipu.ipu_compiler.compile(pipelined_model, inputs=[learning_rate])

    outfeed_op = outfeed_queue.dequeue()

    ipu.utils.move_variable_initialization_to_cpu()
    init_op = tf.global_variables_initializer()

    # Configure the IPU.
    ipu_configuration = ipu.config.IPUConfig()
    ipu_configuration.auto_select_ipus = num_ipus
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
                if (step == (steps-1) or (step % 10) == 0):
                    print("Step {}, Epoch {:.1f}, Mean loss: {:.3f}".format(
                        step, epoch, np.mean(losses)))
        end = time.time()
        elapsed = end - begin
        samples_per_second = training_samples/elapsed
        print("Elapsed {}, {} samples/sec".format(elapsed, samples_per_second))

    print("Program ran successfully")
