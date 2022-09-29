# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_run_demo_fits():
    testing_util.run_command(
        "python full_resnext.py", working_path, ["Throughput", "Latency"]
    )
