# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import inspect
import json
import os
import pytest
import subprocess
import unittest
from tempfile import TemporaryDirectory

import tutorials_tests.testing_util as testing_util


def run_pipelining_example(py_args):
    """Helper function to run the pipelining example"""
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    cmd = ['python', 'pipelining.py']
    args = [str(item) for sublist in py_args.items()
            for item in sublist if item != ""]
    cmd.extend(args)
    try:
        out = subprocess.check_output(
            cmd, cwd=cwd, stderr=subprocess.PIPE, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out


class TestPipeliningTensorFlow(unittest.TestCase):
    """Tests for the pipelining TensorFlow code example"""

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_pipelining_convergence(self):
        """Run with default settings and check it converges"""
        out = run_pipelining_example({})
        # Get the final loss
        loss_regex = r"loss: ([\d.]+)"
        result = testing_util.parse_results_with_regex(out, loss_regex)
        # Get the last loss
        loss = result[0][-1]
        self.assertGreater(loss, 0.001)
        self.assertLess(loss, 0.02)



if __name__ == "__main__":
    unittest.main()
