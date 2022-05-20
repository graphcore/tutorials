# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import pytest
import os
import sys
import re
import unittest

from pathlib import Path

from tutorials_tests import testing_util


# Mark it as using all the IPUs to avoid other tests from interfering.
# With other tests running in parallel there is a race condition in
# which another test could acquire a device between the parent
# device configuration (by poprun) and the child device acquisition
# (by the instances).
@pytest.mark.category1
@pytest.mark.ipus(16)
class TestPopDistExample:
    script_dir = Path(os.path.abspath(__file__)).parents[1]
    accuracy_tolerance = 0.03

    def test_training_then_inference(self, tmpdir):
        self.run_script_and_check_loss(num_instances=2, num_total_replicas=4, script="popdist_training.py", expected_acc=0.7588, working_dir=tmpdir)
        self.run_script_and_check_loss(num_instances=4, num_total_replicas=4, script="popdist_inference.py", expected_acc=0.7731, working_dir=tmpdir)

    def test_training_1_replica_per_instance(self, tmpdir):
        self.run_script_and_check_loss(num_instances=2, num_total_replicas=2, script="popdist_training.py", expected_acc=0.8228, working_dir=tmpdir)

    def run_script_and_check_loss(self, num_instances, num_total_replicas, script, expected_acc, working_dir):
        cmd = [
            "poprun",
            "--num-replicas", str(num_total_replicas),
            "--num-instances", str(num_instances),
            sys.executable,
            str(self.script_dir / script)
        ]

        logging.info(f"Executing: {cmd} in {working_dir}")
        out = testing_util.run_command_fail_explicitly(cmd, cwd=working_dir)
        acc = float(re.findall(r"accuracy: \d+\.\d+", out)[-1].split(" ")[-1])
        assert abs(expected_acc - acc) < self.accuracy_tolerance, (
            f"Measured accuracy {acc} does fall within the "
            f"{self.accuracy_tolerance} tolerance around the expected "
            f"accuracy of {expected_acc}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
