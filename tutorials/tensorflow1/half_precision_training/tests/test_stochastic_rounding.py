# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from tutorials_tests.testing_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestStochasticRounding(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_run_complete_ipu_fp16(self):
        """ Check ResNet-8 runs in float-16 """
        self.run_command("python stochastic_rounding.py float16 8",
                         working_path,
                         "Program ran successfully")

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_run_complete_ipu_fp32(self):
        """ Check ResNet-8 runs in float-32 """
        self.run_command("python stochastic_rounding.py float32 8",
                         working_path,
                         "Program ran successfully")
