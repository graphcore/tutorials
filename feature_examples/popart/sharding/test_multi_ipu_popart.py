# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import sys

import pytest
import tutorials_tests.testing_util as testing_util


def run_multi_ipu(shards, batch_size, batches_per_step):
    py_version = "python" + str(sys.version_info[0])
    cmd = [py_version, "multi_ipu.py",
           "--shards", str(shards),
           "--batch-size", str(batch_size),
           "--batches-per-step", str(batches_per_step)]

    out = testing_util.run_command_fail_explicitly(cmd, os.path.dirname(__file__))

    return out



"""Tests for multi-IPU popART code example"""


# Multi-IPU tests
@pytest.mark.ipus(2)
@pytest.mark.category1
def test_multi_ipu_2_10():
    out = run_multi_ipu(shards=2, batch_size=10, batches_per_step=100000)
