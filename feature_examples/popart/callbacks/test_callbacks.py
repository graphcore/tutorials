# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import pytest
import os
import tutorials_tests.testing_util as testing_util


"""Tests for the popART LSTM synthetic benchmarks"""


@pytest.mark.category1
def test_example_runs():
    working_path = os.path.dirname(__file__)
    testing_util.run_command(
        "python3 callbacks.py --data-size 1000", working_path, ["Mul:0", "Add:0"]
    )
