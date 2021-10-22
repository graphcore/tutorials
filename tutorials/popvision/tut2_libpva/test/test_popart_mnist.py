# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import pathlib
import unittest
from tempfile import TemporaryDirectory

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper, parse_results_for_accuracy

# Variable which points to the root directory of this tutorial
TUT2LIBPVA_ROOT_PATH = pathlib.Path(__file__).parents[1]


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

    def check_popart_mnist_profiles(self, absolute_script_directory: str):
        """ Runs the popart_mnist.py and libpva_examples.py scripts stored in the
        directory passed to this method. The directory needs to be absolute to ensure
        the script files will be where expected regardless of running location.
        """
        # Build the absolute paths to the scripts
        popart_script = os.path.join(
            absolute_script_directory, "popart_mnist.py")
        libpva_script = os.path.join(
            absolute_script_directory, "libpva_examples.py")

        with TemporaryDirectory(dir=os.path.dirname(__file__)) as cache_dir:
            os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true",'\
                '"autoReport.directory":"' + cache_dir + '"}'
            py_args = self.generic_arguments.copy()
            out = run_python_script_helper(
                os.path.dirname(__file__), popart_script, **py_args)
            expected_accuracy = [88.88]
            parse_results_for_accuracy(
                out, expected_accuracy, self.accuracy_tolerances
            )

            # Verify that the profile.pop file is created in both of the training and inference
            # subdirectories.
            for sub_folder in ["training", "inference"]:
                profile_dir = os.path.join(cache_dir, sub_folder)
                if not any(fname.endswith(".pop") for fname in os.listdir(profile_dir)):
                    raise AssertionError(
                        "No {}/profile.pop file found".format(sub_folder))
                run_python_script_helper(profile_dir, libpva_script)

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_setup(self):
        """Tests that the model in the `start_here` directory trains and a
        profile.pop file is created"""
        start_here_directory = (TUT2LIBPVA_ROOT_PATH / "start_here").absolute()
        self.check_popart_mnist_profiles(str(start_here_directory))

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_complete(self):
        """Tests that the model in the `complete` directory trains and a
        profile.pop file is created"""
        complete_directory = (TUT2LIBPVA_ROOT_PATH / "complete").absolute()
        self.check_popart_mnist_profiles(str(complete_directory))
