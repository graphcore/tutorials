# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path

import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


"""High-level integration tests for TensorFlow grouped convolution synthetic benchmarks"""


@pytest.mark.category1
def test_help():
    testing_util.run_command(
        "python3 grouped_conv.py --help", working_path, "usage: grouped_conv.py"
    )


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_default():
    testing_util.run_command(
        "python3 grouped_conv.py", working_path, [r"(\w+.\w+) items/sec"]
    )


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_inference():
    testing_util.run_command(
        "python3 grouped_conv.py --batch-size 8 --use-generated-data",
        working_path,
        [r"(\w+.\w+) items/sec"],
    )


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_block_repeats_and_group_dims():
    testing_util.run_command(
        "python3 grouped_conv.py --block-repeats 20 --group-dim 8",
        working_path,
        [r"(\w+.\w+) items/sec"],
    )


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_training():
    testing_util.run_command(
        "python3 grouped_conv.py --train --input-size 112  --stride 3 --filter-in 32 --filter-out 16",
        working_path,
        [r"(\w+.\w+) items/sec", "Input size 112"],
    )


@pytest.mark.category1
@pytest.mark.ipus(2)
def test_replicas():
    testing_util.run_command(
        "python3 grouped_conv.py --replicas 2",
        working_path,
        [r"(\w+.\w+) items/sec"],
    )
