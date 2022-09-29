# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path

import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


"""High-level integration tests for TensorFlow Dense layer synthetic benchmarks"""


@pytest.mark.category1
def test_help():
    testing_util.run_command("python3 dense.py --help", working_path, "usage: dense.py")


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_default():
    testing_util.run_command("python3 dense.py", working_path, [r"(\w+.\w+) items/sec"])


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_train_with_activation():
    testing_util.run_command(
        "python3 dense.py --train --include-activation --size 256 --batch-size 128",
        working_path,
        [r"(\w+.\w+) items/sec"],
    )


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_convolution_options():
    testing_util.run_command(
        'python3 dense.py --convolution-options={"availableMemoryProportion":"0.2"} --steps 1',
        working_path,
        [r"(\w+.\w+) items/sec"],
    )


@pytest.mark.category1
@pytest.mark.ipus(2)
def test_replicas():
    testing_util.run_command(
        "python3 dense.py --replicas 2", working_path, [r"(\w+.\w+) items/sec"]
    )


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_matmul_options():
    testing_util.run_command(
        'python3 dense.py --matmul-options={"partialsType":"half"} --size 128 --batch-size 4',
        working_path,
        [r"(\w+.\w+) items/sec"],
    )
