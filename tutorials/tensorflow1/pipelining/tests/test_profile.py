# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from tutorials_tests.testing_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestProfile(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_run_complete_profile(self):
        """ Check answers/step4_configurable_stages.py works via profiling script """
        self.run_command("scripts/profile.sh answers/step4_configurable_stages.py",
                         working_path,
                         "Program ran successfully")
