# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestStep1(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_run_complete_step1(self):
        """ Check step1_single_ipu.py works """
        self.run_command("python step1_single_ipu.py",
                         working_path,
                         "Program ran successfully")
