# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import subprocess
import sys
import unittest
import pytest

import numpy as np
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.assert_util import assert_result_equals_tensor_value


def run_simple_sharding():
    py_version = "python{}".format(sys.version_info[0])
    cmd = [py_version, "simple_sharding.py"]
    try:
        out = subprocess.check_output(
            cmd, cwd=os.path.dirname(__file__), stderr=subprocess.PIPE, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out


class TestTensorFlowSharding(unittest.TestCase):
    """High-level integration tests for tensorflow sharding examples"""

    @classmethod
    def setUpClass(cls):
        pass

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_manual_sharding(self):
        """Manual sharding example using 2 shards"""
        out = run_simple_sharding()
        assert_result_equals_tensor_value(
            out, np.array([3.0, 8.0], dtype=np.float32)
        )
