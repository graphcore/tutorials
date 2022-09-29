# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parents[1]


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_run_complete_ipu_fp16():
    """Check ResNet-8 runs in float-16"""
    testing_util.run_command(
        "python stochastic_rounding.py float16 8",
        working_path,
        "Program ran successfully",
    )


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_run_complete_ipu_fp32():
    """Check ResNet-8 runs in float-32"""
    testing_util.run_command(
        "python stochastic_rounding.py float32 8",
        working_path,
        "Program ran successfully",
    )
