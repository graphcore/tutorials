# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).resolve().parent.parent


"""Integration tests for Inspecting Tensors example"""


def test_default_commandline():
    """Test the default command line which outfeeds accumulated gradients."""
    testing_util.run_command(
        "python3 pipelined_model.py",
        working_path,
        [
            r"Gradient key: dense2/bias:0_grad shape: \(100, 128\)",
            r"Gradient key: dense2/kernel:0_grad shape: \(100, 256, 128\)",
        ],
    )


def test_model_gradient_accumulation_pre_accumulated_gradients():
    """Test the outfeeding of pre-accumulated gradients."""
    testing_util.run_command(
        "python3 pipelined_model.py --outfeed-pre-accumulated-gradients",
        working_path,
        [
            r"Gradient key: dense2/bias:0_grad shape: \(1600, 128\)",
            r"Gradient key: dense2/kernel:0_grad shape: \(1600, 256, 128\)",
        ],
    )
