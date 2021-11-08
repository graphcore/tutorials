# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import hashlib
import logging
import pytest
import os
import sys
import subprocess
import tempfile
import tensorflow as tf
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
                tempdir, f"instance_{i}.h5") for i in range(NUM_INSTANCES)]
            logging.info(f"Instance checkpoints: {instances}")

            checksums = [hashlib.md5(Path(file).read_bytes()).hexdigest() for file in instances]

            for checksum in checksums:
                self.assertEqual(checksums[0], checksum)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
