# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import pytest
import os
import sys
import subprocess
import tempfile
import tensorflow.compat.v1 as tf
import unittest

from pathlib import Path


# Mark it as using all the IPUs to avoid other tests from interfering.
# With other tests running in parallel there is a race condition in
# which another test could acquire a device between the parent
# device configuration (by poprun) and the child device acquisition
# (by the instances).
@pytest.mark.category1
@pytest.mark.ipus(16)
class TestPopDistTraining(unittest.TestCase):
    def test_instances_in_sync_after_training(self):

        NUM_TOTAL_REPLICAS = 4
        NUM_INSTANCES = 2

        scriptdir = Path(os.path.realpath(__file__)).parent.parent
        cmd = [
            "poprun",
            # The CI runs as root, so let's allow that.
            "--mpi-global-args", "--allow-run-as-root --tag-output",
            "--num-replicas", str(NUM_TOTAL_REPLICAS),
            "--num-instances", str(NUM_INSTANCES),
            sys.executable,
            str(scriptdir / "popdist_training.py")
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            logging.info(f"Executing: {cmd} in {tempdir}")
            subprocess.check_call(cmd, cwd=tempdir)

            # The checkpoint files from all the instances.
            instances = [os.path.join(
                tempdir, f"instance_{i}") for i in range(NUM_INSTANCES)]
            logging.info(f"Instance checkpoints: {instances}")

            # The final weights should be the same on all instances.
            var_names_and_shapes = tf.train.list_variables(instances[0])

            for var_name, var_shape in var_names_and_shapes:
                logging.info(
                    f"Checking variable {var_name} with shape {var_shape}")
                value_instance_0 = tf.train.load_variable(
                    instances[0], var_name)

                for i in range(1, NUM_INSTANCES):
                    value_instance_i = tf.train.load_variable(
                        instances[i], var_name)
                    self.assertListEqual(
                        value_instance_0.tolist(), value_instance_i.tolist())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
