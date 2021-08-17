# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
from pathlib import Path
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class runFileTest(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_program_run(self):
        # Check whether the model compiles and trains
        self.run_command("python3 poptorch_custom_op.py",
                         working_path,
                         "Model using Leaky ReLU custom op")
