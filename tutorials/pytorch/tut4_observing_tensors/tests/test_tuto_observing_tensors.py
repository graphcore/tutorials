# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestComplete(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_run_default_ipu(self):
        # Check default params
        self.run_command("python anchor_tensor_example.py",
                         working_path,
                         "Saved histogram")
