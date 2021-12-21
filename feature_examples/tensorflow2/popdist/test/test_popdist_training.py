# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import hashlib
import logging
import pytest
import os
import sys
import subprocess
import unittest

from pathlib import Path


# Mark it as using all the IPUs to avoid other tests from interfering.
# With other tests running in parallel there is a race condition in
# which another test could acquire a device between the parent
# device configuration (by poprun) and the child device acquisition
# (by the instances).
@pytest.mark.category1
@pytest.mark.ipus(16)
class TestPopDistTraining:
    def test_instances_in_sync_after_training(self, tmpdir):
        NUM_TOTAL_REPLICAS = 4
        NUM_INSTANCES = 2
        self.check_instances_in_sync_after_training(NUM_TOTAL_REPLICAS, NUM_INSTANCES, tmpdir)

    def test_1_replica_per_instance(self, tmpdir):
        NUM_TOTAL_REPLICAS = 2
        NUM_INSTANCES = 2
        self.check_instances_in_sync_after_training(NUM_TOTAL_REPLICAS, NUM_INSTANCES, tmpdir)

    def check_instances_in_sync_after_training(self, NUM_TOTAL_REPLICAS, NUM_INSTANCES, tmpdir):
        scriptdir = Path(os.path.realpath(__file__)).parent.parent
        cmd = [
            "poprun",
            "--num-replicas", str(NUM_TOTAL_REPLICAS),
            "--num-instances", str(NUM_INSTANCES),
            sys.executable,
            str(scriptdir / "popdist_training.py")
        ]

        logging.info(f"Executing: {cmd} in {tmpdir}")
        subprocess.check_call(cmd, cwd=tmpdir)

        # The checkpoint files from all the instances.
        instances = [os.path.join(
            tmpdir, f"instance_{i}.h5") for i in range(NUM_INSTANCES)]
        logging.info(f"Instance checkpoints: {instances}")

        checksums = [hashlib.md5(Path(file).read_bytes()).hexdigest() for file in instances]

        non_matching_instances = [instance for checksum, instance in zip(
            checksums, instances) if checksum != checksums[0]]
        assert not non_matching_instances, (
            "Not all checkpoint files matched " +
            f"{instances[0]}: {non_matching_instances}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
