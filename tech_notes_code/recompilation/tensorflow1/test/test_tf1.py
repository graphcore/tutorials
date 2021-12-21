# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestRecompilation(SubProcessChecker):

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_tf1_recompilation(self):
        self.run_command("python3 recompilation.py",
                         working_path,
                         "Caching/warm up test")
