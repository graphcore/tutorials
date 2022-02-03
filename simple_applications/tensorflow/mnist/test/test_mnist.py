# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
from tutorials_tests.testing_util import SubProcessChecker

working_path = Path(__file__).parent.parent


@pytest.mark.category1
@pytest.mark.ipus(1)
class TensorFlow1Mnist(SubProcessChecker):
    """Integration tests for TensorFlow 1 MNIST example"""

    def test_run_tf1_mnist(self):
        self.run_command("python3 mnist.py",
                         working_path,
                         "Program ran successfully")
