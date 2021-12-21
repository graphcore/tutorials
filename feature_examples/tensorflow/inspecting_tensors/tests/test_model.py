# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from tutorials_tests.testing_util import SubProcessChecker

working_path = Path(__file__).resolve().parent.parent


@pytest.mark.category2
@pytest.mark.ipus(1)
class TestTensorFlowInspectingTensorsNonPipelined(SubProcessChecker):
    """Integration tests for Inspecting Tensors example"""

    def test_default_commandline(self):
        """ Test the default command line which outfeeds accumulated gradients.
        """
        self.run_command("python3 model.py",
                         working_path,
                         [r"Gradient key: FCModel/dense2/bias:0_grad shape: \(100, 128\)",
                          r"Gradient key: FCModel/dense2/kernel:0_grad shape: \(100, 256, 128\)"])


    def test_model_gradient_accumulation_pre_accumulated_gradients(self):
        """ Test the outfeeding of pre-accumulated gradients.
        """
        self.run_command("python3 model.py --repeat-count 1600 --gradient-accumulation-count 16 --outfeed-pre-accumulated-gradients",
                         working_path,
                         [r"Gradient key: FCModel/dense2/bias:0_grad shape: \(1600, 128\)",
                          r"Gradient key: FCModel/dense2/kernel:0_grad shape: \(1600, 256, 128\)"])
