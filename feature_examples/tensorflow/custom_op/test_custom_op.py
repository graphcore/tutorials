# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import re

import numpy as np
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests import testing_util as tu

"""Simple tests for the TensorFlow custom op example"""

path = os.path.dirname(__file__)


@pytest.fixture(autouse=True)
def with_compiled_custom_op():
    """Runs the make command to build the custom op objects"""
    files_to_generate = ["custom_codelet.gp", "libcustom_op.so"]

    if tu.check_data_exists(path, files_to_generate):
        print("Objects already present, cleaning...")
        tu.run_command(["make", "clean"], cwd=path)

    tu.run_command("make", cwd=path)
    if not tu.check_data_exists(path, files_to_generate):
        raise Exception("Custom op compilation failed")

    print("Successfully compiled custom op")


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_tf_code():
    """Run the python script and check the result"""
    # Run the script and capture the output
    out = tu.run_python_script_helper(path, "tf_code.py")

    # Get the first and the second line of output
    ipu_res, target_res = out.split("\n")[:-1]

    # Convert these lines to arrays, in turn
    list_regex = r"\[.*\]$"
    match = re.search(list_regex, ipu_res)
    string_vals = match.group()[1:-1].split()
    ipu_arr = np.array([float(val) for val in string_vals], dtype=np.float32)
    match = re.search(list_regex, target_res)
    string_vals = match.group()[1:-1].split()
    target_arr = np.array([float(val) for val in string_vals], dtype=np.float32)

    # Finally, check that the results are reasonably close
    assert np.allclose(
        ipu_arr, target_arr
    ), f"Output value {ipu_arr} does not equal expected value {target_arr}"

    # Clean up
    tu.run_command(["make", "clean"], cwd=path)
