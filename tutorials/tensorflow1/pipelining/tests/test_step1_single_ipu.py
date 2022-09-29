# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parents[1]


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_run_complete_step1():
    """Check step1_single_ipu.py works"""
    testing_util.run_command(
        "python step1_single_ipu.py", working_path, "Program ran successfully"
    )
