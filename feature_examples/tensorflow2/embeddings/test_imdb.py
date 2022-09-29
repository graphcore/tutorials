# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import pytest
import tutorials_tests.testing_util as testing_util

working_path = os.path.dirname(__file__)


"""Integration tests for TensorFlow 2 IMDB example"""


@pytest.mark.category2
@pytest.mark.ipus(2)
def test_pipeline():
    testing_util.run_command("python imdb.py", working_path, "Epoch 2/")


@pytest.mark.category2
@pytest.mark.ipus(2)
def test_pipeline_sequential():
    testing_util.run_command("python imdb_sequential.py", working_path, "Epoch 2/")


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_single_ipu():
    testing_util.run_command("python imdb_single_ipu.py", working_path, "Epoch 3/")


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_single_ipu_sequential():
    testing_util.run_command(
        "python imdb_single_ipu_sequential.py", working_path, "Epoch 3/"
    )
