# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import os
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests import testing_util


class TestPopARTMNISTImageClassification:
    """High-level integration tests for training with the MNIST data-set"""

    accuracy_tolerances = 3.0
    generic_cmd = ["python", "popart_mnist.py"]
    generic_arguments = {
        "--batch-size": 4,
        "--batches-per-step": 1000,
        "--epochs": 10,
        "--num-ipus": 1,
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
        expected_accuracy = [90.70]
        testing_util.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_sharded(self):
        """Generic test on default arguments in training over 2 IPUs"""
        cmd, args = self.generic_cmd.copy(), self.generic_arguments.copy()
        args["--num-ipus"] = 2
        cmd = testing_util.add_args(cmd, args)
        out = testing_util.run_command_fail_explicitly(cmd, self.cwd)
        expected_accuracy = [90.70]
        testing_util.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_replicated(self):
        """Generic test on default arguments in training over 2 IPUs
        with replication"""
        cmd, args = self.generic_cmd.copy(), self.generic_arguments.copy()
        args["--num-ipus"] = 2
        args["--replication-factor"] = 2
        cmd = testing_util.add_args(cmd, args)
        out = testing_util.run_command_fail_explicitly(cmd, self.cwd)
        expected_accuracy = [90.70]
        testing_util.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_sharded_pipelined(self):
        """Generic test on default arguments in training over 2 IPUs
        and pipelined"""
        cmd, args = self.generic_cmd.copy(), self.generic_arguments.copy()
        args["--num-ipus"] = 2
        args["--pipeline"] = None
        cmd = testing_util.add_args(cmd, args)
        out = testing_util.run_command_fail_explicitly(cmd, self.cwd)
        expected_accuracy = [89.25]
        testing_util.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(4)
    @pytest.mark.category2
    def test_mnist_train_replicated_pipelined(self):
        """Generic test on default arguments in training over 2 IPUs
        and pipelined"""
        cmd, args = self.generic_cmd.copy(), self.generic_arguments.copy()
        args["--num-ipus"] = 4
        args["--replication-factor"] = 2
        args["--pipeline"] = None
        cmd = testing_util.add_args(cmd, args)
        out = testing_util.run_command_fail_explicitly(cmd, self.cwd)
        expected_accuracy = [89.25]
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
    def test_mnist_simulation(self):
        """Simulation test with basic arguments"""
        cmd, args = self.generic_cmd.copy(), self.generic_arguments.copy()
        args["--epochs"] = 1
        args["--simulation"] = None
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
