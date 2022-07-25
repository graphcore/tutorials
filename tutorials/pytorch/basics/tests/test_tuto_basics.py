# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_model():
    testing_util.run_command("python3 walkthrough.py", working_path, "Eval accuracy:")
