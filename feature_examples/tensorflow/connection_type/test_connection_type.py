# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import os
import pytest
from more_itertools import locate
from tempfile import TemporaryDirectory

# NOTE: The imports below are dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests.testing_util import run_python_script_helper
from tutorials_tests.assert_util import assert_result_equals_tensor_value


# Strings used to identify lines from trace that
# indicate graph compilation and device attachment respectively.
# Using trace here rather than events because it is less intrusive
# and avoids polluting the example itself with unnecessary complexity.
COMPILE_STRING = "Compiled "
ATTACH_STRING = "attached to "
PRE_COMPILE_STRING = "pre-compiled "


def parse_output(out):
    """Helper to parse output (stdout/stderr) and return a
    dictionary that includes the result plus line indices
    for compilation and attachments."""
    lines = out.splitlines()
    compile_list = list(locate(lines, lambda l: COMPILE_STRING in l))
    attach_list = list(locate(lines, lambda l: ATTACH_STRING in l))
    pre_compile_list = list(locate(lines, lambda l: PRE_COMPILE_STRING in l))
    return {
        "result": lines[-1],
        "compile": compile_list,
        "attach": attach_list,
        "pre_compile": pre_compile_list,
    }


def run_connection_type(connection_type):
    """Helper to run connect_type.py with specific connection type,
    capture the output, and parse the result."""
    kwargs = {"--connection_type": connection_type}
    out = run_python_script_helper(
        os.path.dirname(__file__), "connection_type.py", want_std_err=True, **kwargs
    )
    result = parse_output(out)
    print(f"result {result}")
    return result


"""High-level integration tests for tensorflow connection type examples"""


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_connection_type_always():
    """Connection type ALWAYS"""
    result = run_connection_type("ALWAYS")
    # Assert correct result.
    assert_result_equals_tensor_value(
        result["result"], np.array([3.0, 8.0], dtype=np.float32)
    )
    # Assert single occurrences of attach and compile
    # with attach occurring first.
    assert len(result["attach"]) == 1, "Missing attach"
    assert len(result["compile"]) == 1, "Missing compile"
    assert result["attach"][0] < result["compile"][0], "Compile before attach"


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_connection_type_on_demand():
    """Connection type ON_DEMAND"""
    result = run_connection_type("ON_DEMAND")
    # Assert correct result.
    assert_result_equals_tensor_value(
        result["result"], np.array([3.0, 8.0], dtype=np.float32)
    )
    # Assert single occurrences of attach and compile
    # with compilation occurring first.
    assert len(result["attach"]) == 1, "Missing attach"
    assert len(result["compile"]) == 1, "Missing compile"
    assert result["attach"][0] > result["compile"][0], "Compile after attach"


@pytest.mark.category1
def test_connection_type_never():
    """Connection type NEVER"""
    result = run_connection_type("NEVER")

    assert result["result"] == "Compiled"

    # Assert single occurrence of compile without attach.
    assert len(result["attach"]) == 0, "Unexpected attach"
    assert len(result["compile"]) == 1, "Missing compile"


@pytest.mark.category1
def test_connection_type_pre_compile():
    """Connection type PRE_COMPILE"""
    with TemporaryDirectory() as cache_dir:
        os.environ["TF_POPLAR_FLAGS"] = "--executable_cache_path=" + cache_dir
        result = run_connection_type("PRE_COMPILE")
        # Assert empty array
        assert_result_equals_tensor_value(
            result["result"], np.array([0.0, 0.0], dtype=np.float32)
        )
        # Assert single occurrence of compile without attach.
        assert len(result["attach"]) == 0, "Unexpected attach"
        assert len(result["pre_compile"]) == 1, "Missing pre-compile"
