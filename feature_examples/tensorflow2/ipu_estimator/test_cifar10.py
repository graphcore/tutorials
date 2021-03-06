# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import subprocess
import unittest
import pytest
from tutorials_tests.testing_util import parse_results_for_accuracy
import tutorials_tests.testing_util as testing_util


def run_cifar10(file_name):
    cwd = os.path.dirname(os.path.abspath(__file__))
    cmd = ["python3", 'cifar10_{0}.py'.format(file_name)]
    out = testing_util.run_command_fail_explicitly(cmd, cwd)

    parse_results_for_accuracy(out, [81.0], 6.0)
    return out


class TestCIFAR10(unittest.TestCase):

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_ipuestimator(self):
        run_cifar10("ipu_estimator")

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_replica(self):
        run_cifar10("ipu_estimator_replica")

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_pipeline(self):
        run_cifar10("ipu_pipeline_estimator")
