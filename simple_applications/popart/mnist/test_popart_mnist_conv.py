# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests import testing_util


class TestPopARTMNISTImageClassificationConvolution:
    """High-level integration tests for training with the MNIST data-set"""

    accuracy_tolerances = 3.0
    generic_cmd = ["python", "popart_mnist_conv.py"]
    generic_arguments = {
        "--batch-size": 4,
        "--batches-per-step": 1000,
        "--epochs": 10,
        "--validation-final-epoch": None,
    }
    cwd = os.path.dirname(os.path.abspath(__file__))

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_train(self):
        """Generic test on default arguments in training"""
        cmd, args = self.generic_cmd.copy(), self.generic_arguments.copy()
        cmd = testing_util.add_args(cmd, args)
        out = testing_util.run_command_fail_explicitly(cmd, self.cwd)
        expected_accuracy = [98.41]
        testing_util.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_all_data(self):
        """Generic test using all the available data (10,000)"""
        cmd, args = self.generic_cmd.copy(), self.generic_arguments.copy()
        args["--epochs"] = 2
        args["--batch-size"] = 10
        args["--batches-per-step"] = 1000
        cmd = testing_util.add_args(cmd, args)
        testing_util.run_command_fail_explicitly(cmd, self.cwd)

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_log_graph_trace(self):
        """Basic test with log-graph-trace argument"""
        cmd, args = self.generic_cmd.copy(), self.generic_arguments.copy()
        args["--epochs"] = 1
        args["--log-graph-trace"] = None
        cmd = testing_util.add_args(cmd, args)
        testing_util.run_command_fail_explicitly(cmd, self.cwd)

    @pytest.mark.category3
    def test_mnist_conv_simulation(self):
        """Simulation test with basic arguments - This simulation takes
           around 838s (~14 minutes) to complete"""
        cmd, args = self.generic_cmd.copy(), self.generic_arguments.copy()
        args["--batch-size"] = 1
        args["--batches-per-step"] = 1
        args["--epochs"] = 1
        args["--simulation"] = None
        cmd = testing_util.add_args(cmd, args)
        testing_util.run_command_fail_explicitly(cmd, self.cwd)
