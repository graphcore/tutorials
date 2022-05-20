# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
from pathlib import Path
import pytest
from common import download_mnist

# NOTE: The imports below are dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests.testing_util import SubProcessChecker
from tutorials_tests.xdist_util import lock


working_path = Path(__file__).parent.parent


class TestBuildAndRun(SubProcessChecker):

    @lock(os.path.join(working_path, "tests/binary.lock"))
    def setUp(self):
        download_mnist(working_path)
        self.run_command("make all", working_path, [])


    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_ipu_model(self):
        ''' Check that the tutorial code runs on the IPU Model '''
        self.run_command("./regression-demo 1 1.0",
                         working_path,
                         ["Using the IPU Model", "Epoch", "100%"])


    @pytest.mark.category2
    def test_ipu_hardware(self):
        ''' Check that the tutorial code runs on the IPU hardware '''
        self.run_command("./regression-demo -IPU 1 1.0",
                         working_path,
                         ["Using the IPU", "Epoch", "100%"])
