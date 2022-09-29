# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests.testing_util import (
    parse_results_for_accuracy,
    run_python_script_helper,
)

# Variable which points to the root directory of this tutorial
TUT_LIBPVA_ROOT_PATH = Path(__file__).parents[1]


"""High-level integration tests for training with the MNIST data-set"""

accuracy_tolerances = 3.0
generic_arguments = {
    "--batch-size": 4,
    "--batches-per-step": 1000,
    "--epochs": 1,
    "--num-ipus": 1,
}


def check_popart_mnist_profiles(absolute_script_directory: Path):
    """Runs the popart_mnist.py and libpva_examples.py scripts stored in the
    directory passed to this method. The directory needs to be absolute to ensure
    the script files will be where expected regardless of running location.
    """
    # Build the absolute paths to the scripts
    popart_script = absolute_script_directory / "popart_mnist.py"
    libpva_script = absolute_script_directory / "libpva_examples.py"
    with TemporaryDirectory(dir=os.path.dirname(__file__)) as cache_dir:
        os.environ["POPLAR_ENGINE_OPTIONS"] = (
            '{"autoReport.all":"true",' '"autoReport.directory":"' + cache_dir + '"}'
        )
        py_args = generic_arguments.copy()
        out = run_python_script_helper(
            os.path.dirname(__file__), str(popart_script), **py_args
        )
        expected_accuracy = [88.88]
        parse_results_for_accuracy(out, expected_accuracy, accuracy_tolerances)
        # Verify that the profile.pop file is created in both of the training and inference
        # subdirectories.
        for sub_folder in ["training", "inference"]:
            profile_dir = os.path.join(cache_dir, sub_folder)
            if not any(fname.endswith(".pop") for fname in os.listdir(profile_dir)):
                raise AssertionError(f"No {sub_folder}/profile.pop file found")
            run_python_script_helper(profile_dir, str(libpva_script))


@pytest.mark.ipus(1)
@pytest.mark.category2
def test_mnist_setup():
    """Tests that the model in the `start_here` directory trains and a
    profile.pop file is created"""
    start_here_directory = (TUT_LIBPVA_ROOT_PATH / "start_here").absolute()
    check_popart_mnist_profiles(start_here_directory)


@pytest.mark.ipus(1)
@pytest.mark.category2
def test_mnist_complete():
    """Tests that the model in the `complete` directory trains and a
    profile.pop file is created"""
    complete_directory = (TUT_LIBPVA_ROOT_PATH / "complete").absolute()
    check_popart_mnist_profiles(complete_directory)
