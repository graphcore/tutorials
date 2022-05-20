# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
from tutorials_tests.testing_util import run_command_fail_explicitly

cwd = Path(__file__).parent.parent


class TestParallelism():

    @pytest.mark.category1
    @pytest.mark.ipus(4)
    def test_sharding(self):
        cmd = ["python", "graph_sharding.py"]
        out = run_command_fail_explicitly(cmd, cwd)

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_inference(self):
        cmd = ["python", "inference_pipelining.py"]
        out = run_command_fail_explicitly(cmd, cwd)

    @pytest.mark.category1
    @pytest.mark.ipus(4)
    def test_training(self):
        cmd = ["python", "training_pipelining.py"]
        out = run_command_fail_explicitly(cmd, cwd)
