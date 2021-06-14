# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestFloat32MasterWeights(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_run_complete_ipu_mixed(self):
        """ Check the program runs in mixed precision """
        self.run_command("python float32_master_weights.py mixed",
                         working_path,
                         "Program ran successfully")

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_run_complete_ipu_fp32(self):
        """ Check the program runs in pure float-32 """
        self.run_command("python float32_master_weights.py float32",
                         working_path,
                         "Program ran successfully")
