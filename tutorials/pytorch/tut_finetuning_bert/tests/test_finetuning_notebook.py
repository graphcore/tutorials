# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import shutil
from pathlib import Path
import pytest
from examples_tests.test_util import SubProcessChecker


class TestComplete(SubProcessChecker):

    @pytest.mark.category3
    @pytest.mark.ipus(16)
    def test_model(self):
        working_path = Path(__file__).parent.parent

        # Run notebook and check that it runs correctly
        cmd = "ipython Fine-tuning-BERT.py"
        self.run_command(cmd,
                         working_path,
                         "Question: What speed-up can one expect from using sequence packing for training BERT on IPU?")
        shutil.rmtree(working_path/"checkpoints")
        shutil.rmtree(working_path/"exe_cache")
