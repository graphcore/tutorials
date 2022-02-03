# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import unittest
import pathlib
import pytest
import tensorflow as tf
from tutorials_tests.testing_util import SubProcessChecker

working_path = pathlib.Path(__file__).parents[1]


@pytest.mark.category1
@pytest.mark.ipus(1)
class TensorFlow2Mnist(SubProcessChecker):
    """Integration tests for TensorFlow 2 MNIST example"""

    @unittest.skipIf(tf.__version__[0] != '2', "Needs TensorFlow 2")
    def test_default_commandline(self):
        self.run_command("python3 mnist_code_only.py",
                         working_path,
                         "Epoch 2/")
