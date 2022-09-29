# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

import utils
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import image_ops

# values taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L198
NORMALIZED_MEAN = [0.485, 0.456, 0.406]
NORMALIZED_STD = [0.229, 0.224, 0.225]


class ResNet(object):
    def __init__(self, image, num_classes):
        self._image = image
        self._num_classes = num_classes
        self._counted_scope = []
        self._flops = 0
        self._weights = 0

    def build_tower(self, image):
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 1, 2, 2, 2]

        # This is an optimisation used in the CNNs training example. It performs
        # data normalisation and casting on the IPU and also pads the image to
        # add a 4th channel. For more information, see:
        # https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/3.0.0/tensorflow/api.html#tensorflow.python.ipu.image_ops.normalise_image
        with tf.variable_scope("normalise_image"):
            scale = 1.0 / 255.0
            mean = NORMALIZED_MEAN
            std = NORMALIZED_STD
            image = image_ops.normalise_image(image, mean, std, scale)

        # conv1
        with tf.variable_scope("conv1"):
            x = self._conv(image, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")

        # conv2_x
        x = self._residual_block_first(x, filters[1], strides[1], name="conv2_1")
        x = self._residual_block(x, name="conv2_2")

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name="conv3_1")
        x = self._residual_block(x, name="conv3_2")

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name="conv4_1")
        x = self._residual_block(x, name="conv4_2")

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name="conv5_1")
        x = self._residual_block(x, name="conv5_2")

        # Logit
        with tf.variable_scope("logits") as scope:
            x = tf.reduce_mean(x, [1, 2])
            x = self._fc(x, self._num_classes)

        # Prob
        probs = tf.nn.softmax(x)

        return probs

    def build_model(self):
        self.probs = ipu_compiler.compile(self.build_tower, inputs=[self._image])

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:

            # Shortcut connection
            shortcut = self._conv(x, 1, out_channel, strides, name="shortcut_conv")
            shortcut = self._bn(shortcut, name="shortcut_norm")
            # Residual
            x = self._conv(x, 3, out_channel, strides, name="conv_1")
            x = self._bn(x, name="bn_1")
            x = self._relu(x, name="relu_1")
            x = self._conv(x, 3, out_channel, 1, name="conv_2")
            x = self._bn(x, name="bn_2")
            # Merge
            x = x + shortcut
            x = self._relu(x, name="relu_2")
        return x

    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(
                x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name="conv_1"
            )
            x = self._bn(x, name="bn_1")
            x = self._relu(x, name="relu_1")
            x = self._conv(
                x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name="conv_2"
            )
            x = self._bn(x, name="bn_2")

            x = x + shortcut
            x = self._relu(x, name="relu_2")
        return x

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    # Helper functions(counts FLOPs and number of weights)
    def _conv(
        self,
        x,
        filter_size,
        out_channel,
        stride,
        pad="SAME",
        input_q=None,
        output_q=None,
        name="conv",
    ):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(
            x, filter_size, out_channel, stride, pad, input_q, output_q, name
        )
        f = (
            2
            * (h / stride)
            * (w / stride)
            * in_channel
            * out_channel
            * filter_size
            * filter_size
        )
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, False, name)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)
