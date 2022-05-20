# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
from tutorials_tests.testing_util import SubProcessChecker


working_path = Path(__file__).parent.parent


class TestComplete(SubProcessChecker):

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_run_demo_fits(self):
        self.run_command("python full_resnext.py",
                         working_path, ['Throughput', 'Latency'])
