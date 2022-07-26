# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import sys
import pytest

import numpy as np

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util
from tutorials_tests.assert_util import assert_result_equals_tensor_value


def run_simple_sharding():
    py_version = "python{}".format(sys.version_info[0])
    cmd = [py_version, "simple_sharding.py"]
    out = testing_util.run_command_fail_explicitly(cmd, os.path.dirname(__file__))
    return out


"""High-level integration tests for tensorflow sharding examples"""


@pytest.mark.ipus(2)
@pytest.mark.category1
def test_manual_sharding():
    """Manual sharding example using 2 shards"""
    out = run_simple_sharding()
    assert_result_equals_tensor_value(out, np.array([3.0, 8.0], dtype=np.float32))
