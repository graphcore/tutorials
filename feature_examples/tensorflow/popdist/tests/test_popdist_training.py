# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import pytest
import os
import sys
import tempfile
import tensorflow.compat.v1 as tf
from tutorials_tests import testing_util

from pathlib import Path


# Mark it as using all the IPUs to avoid other tests from interfering.
# With other tests running in parallel there is a race condition in
# which another test could acquire a device between the parent
# device configuration (by poprun) and the child device acquisition
# (by the instances).
@pytest.mark.category1
@pytest.mark.ipus(16)
class TestPopDistTraining:
    @pytest.mark.parametrize("num_replicas", [2, 4])
    def test_instances_in_sync_after_training(self, num_replicas):
        with tempfile.TemporaryDirectory() as cachedir:
            for num_instances in [1, 2, 4]:
                if num_instances > num_replicas:
                    continue
                scriptdir = Path(os.path.realpath(__file__)).parent.parent
                cmd = [
                    "poprun",
                    "-vv",
                    # The CI runs as root, so let's allow that.
                    "--mpi-global-args",
                    "--tag-output",
                    "--num-replicas",
                    str(num_replicas),
                    "--num-instances",
                    str(num_instances),
                    "--executable-cache-path",
                    str(cachedir),
                    sys.executable,
                    str(scriptdir / "popdist_training.py"),
                ]

                with tempfile.TemporaryDirectory() as workdir:
                    logging.info(f"Executing: {cmd} in {workdir}")
                    testing_util.run_command_fail_explicitly(cmd, workdir)

                    # The checkpoint files from all the instances.
                    instance_checkpoints = [
                        os.path.join(workdir, f"instance_{i}")
                        for i in range(num_instances)
                    ]

                    # The final weights should be the same on all instances.
                    self._assert_all_variables_are_equal(instance_checkpoints)

    def _assert_all_variables_are_equal(self, instance_checkpoints):
        logging.info(f"Checking instance checkpoints: {instance_checkpoints}")
        var_names_and_shapes = tf.train.list_variables(instance_checkpoints[0])

        for var_name, var_shape in var_names_and_shapes:
            logging.info(f"Checking variable {var_name} with shape {var_shape}")

            first_value = tf.train.load_variable(instance_checkpoints[0], var_name)

            for other_checkpoint in instance_checkpoints[1:]:
                other_value = tf.train.load_variable(other_checkpoint, var_name)
                assert (
                    first_value.tolist() == other_value.tolist()
                ), f"Variable {var_name} did not match."


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pytest.main()
