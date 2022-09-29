# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


"""Integration tests for TensorFlow 1 MNIST example"""


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_run_tf1_mnist():
    testing_util.run_command(
        "python3 mnist.py", working_path, "Program ran successfully"
    )
