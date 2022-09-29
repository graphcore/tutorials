# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from pathlib import Path

import pytest
from filelock import FileLock
from tutorials_tests import testing_util

build_dir = Path(__file__).parent


@pytest.fixture(autouse=True)
def with_compiled_example():
    with FileLock(__file__ + ".lock"):
        testing_util.run_command("make", build_dir)


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_run_regression():
    testing_util.run_command(
        "python3 regression.py", build_dir, ["Losses, grads and weights match."]
    )
