# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from tutorials_tests.testing_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestStep3(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_run_complete_step3(self):
        """ Check answers/step3_pipelining.py works """
        self.run_command("python answers/step3_pipelining.py",
                         working_path,
                         "Program ran successfully")
