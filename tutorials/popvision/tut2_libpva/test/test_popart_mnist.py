# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import unittest
from tempfile import TemporaryDirectory

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper, run_test_helper, \
    parse_results_for_accuracy


def run_popart_mnist_setup(**kwargs):
    """Helper function to run popart mnist linear model python script with
       command line arguments"""
    out = run_python_script_helper(os.path.dirname(__file__), "../start_here/popart_mnist.py",
                                   **kwargs)
    return out


def run_popart_mnist_complete(**kwargs):
    """Helper function to run popart mnist linear model python script with
       command line arguments"""
    out = run_python_script_helper(os.path.dirname(__file__), "../complete/popart_mnist.py",
                                   **kwargs)
    return out


class TestPopVisionAnalysis(unittest.TestCase):
    """High-level integration tests for training with the MNIST data-set"""

    @classmethod
    def setUpClass(cls):
        cls.accuracy_tolerances = 3.0
        cls.generic_arguments = {
            "--batch-size": 4,
            "--batches-per-step": 1000,
            "--epochs": 1,
            "--num-ipus": 1
        }

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_setup(self):
        """Tests that the model trains and a profile.pop file is created"""
        with TemporaryDirectory(dir = os.path.dirname(__file__)) as cache_dir:
            os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true",'\
                '"autoReport.directory":"' + cache_dir + '"}'
            py_args = self.generic_arguments.copy()
            out = run_test_helper(
                run_popart_mnist_setup,
                **py_args
            )
            expected_accuracy = [88.88]
            parse_results_for_accuracy(
                out, expected_accuracy, self.accuracy_tolerances
            )

            # Verify that the profile.pop file is created
            if not any(fname.endswith(".pop") for fname in os.listdir(cache_dir)):
                raise AssertionError("No profile.pop file found")

            run_python_script_helper(cache_dir, "../../start_here/libpva_examples.py")

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_complete(self):
        """Tests that the model trains and a profile.pop file is created"""
        with TemporaryDirectory(dir = os.path.dirname(__file__)) as cache_dir:
            os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true",'\
                '"autoReport.directory":"' + cache_dir + '"}'
            py_args = self.generic_arguments.copy()
            out = run_test_helper(
                run_popart_mnist_complete,
                **py_args
            )
            expected_accuracy = [88.88]
            parse_results_for_accuracy(
                out, expected_accuracy, self.accuracy_tolerances
            )
            # Verify that the profile.pop file is created
            if not any(fname.endswith(".pop") for fname in os.listdir(cache_dir)):
                raise AssertionError("No profile.pop file found")

            run_python_script_helper(cache_dir, "../../complete/libpva_examples.py")
