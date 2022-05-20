""" ​
Copyright (c) 2022 Graphcore Ltd. All rights reserved. ​
"""

from pathlib import Path
import pytest

from tutorials_tests.testing_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestDemo(SubProcessChecker):
    @pytest.mark.category2
    def test_run_demo(self):
        self.run_command("python3 demo.py",
                         working_path,
                         "128/128")
