# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parents[1]


@pytest.mark.category2
@pytest.mark.ipus(2)
def test_run_ipu_pipeline_estimator():
    """Check answers/ipu_pipeline_estimator.py works"""
    testing_util.run_command(
        "python answers/ipu_pipeline_estimator.py",
        working_path,
        "Program ran successfully",
    )
