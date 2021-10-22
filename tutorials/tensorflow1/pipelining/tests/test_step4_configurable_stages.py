# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestStep4(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_run_complete_step4(self):
        """ Check answers/step4_configurable_stages.py works """
        self.run_command("python answers/step4_configurable_stages.py",
                         working_path,
                         "Program ran successfully")
