# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import tutorials_tests.testing_util as testing_util
import pathlib
import os
import sys
import pytest
import filelock
from subprocess import run

working_directory = pathlib.Path(__file__).parents[1].resolve()


@pytest.mark.usefixtures("custom_ops")
@pytest.mark.category2
@pytest.mark.ipus(2)
def test_small_embedding(tmp_path):
    out = testing_util.run_command_fail_explicitly(
        [sys.executable, "tests/sharded_embedding_tool.py", "--ipus", "2"],
        working_directory,
    )
    assert "Results match." in out


@pytest.mark.usefixtures("custom_ops")
@pytest.mark.category2
@pytest.mark.ipus(4)
def test_larger_embedding(tmp_path):
    out = testing_util.run_command_fail_explicitly(
        [
            sys.executable,
            "tests/sharded_embedding_tool.py",
            "--ipus",
            "4",
            "--vocab-size",
            "8000",
            "--feature-size",
            "768",
            "--sequence-length",
            "256",
        ],
        working_directory,
    )
    assert "Results match." in out


@pytest.fixture(scope="session")
def custom_ops():
    """This function builds the ipu_sparse_ops
    library for any tests that rely on it.
    """
    build_path = pathlib.Path(__file__).parents[1]

    shared_libs = ["libconcurrent_ops.so"]
    paths = [pathlib.Path(build_path, "custom_ops", f) for f in shared_libs]

    # Use exclusive lockfile to avoid race conditions on the build:
    lock_path = pathlib.Path(build_path, ".custom_ops.pytest.build.lockfile")

    print(f"Building paths: {paths}")

    def build_ops():
        run(["make", "-j"], cwd=build_path)

    with filelock.FileLock(lock_path):
        build_ops()
