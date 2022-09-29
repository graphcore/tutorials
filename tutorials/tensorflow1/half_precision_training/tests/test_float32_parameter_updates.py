# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parents[1]


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_run_complete_ipu_mixed():
    """Check the program runs in mixed precision"""
    testing_util.run_command(
        "python float32_parameter_updates.py mixed",
        working_path,
        "Program ran successfully",
    )


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_run_complete_ipu_fp32():
    """Check the program runs in pure FP32"""
    testing_util.run_command(
        "python float32_parameter_updates.py float32",
        working_path,
        "Program ran successfully",
    )
