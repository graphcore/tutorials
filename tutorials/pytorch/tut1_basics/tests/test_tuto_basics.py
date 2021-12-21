# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from tutorials_tests.testing_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestComplete(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_model(self):
        # Check whether the model compiles and executes.
        self.run_command("python3 walkthrough.py",
                         working_path,
                         "Eval accuracy:")
