# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import sys

import numpy as np
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests.testing_util import run_command_fail_explicitly
from tutorials_tests.assert_util import assert_result_equals_tensor_value


def run_simple_sharding():
    py_version = f"python{sys.version_info[0]}"
    cmd = [py_version, "simple_sharding.py"]
    out = run_command_fail_explicitly(
        cmd, os.path.dirname(__file__), suppress_warnings=True
    )
    return out


"""High-level integration tests for tensorflow sharding examples"""


@pytest.mark.ipus(2)
@pytest.mark.category1
def test_manual_sharding():
    """Manual sharding example using 2 shards"""
    out = run_simple_sharding()
    assert_result_equals_tensor_value(out, np.array([3.0, 8.0], dtype=np.float32))
