# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests.testing_util import SubProcessChecker
from tutorials_tests.xdist_util import lock

working_path = Path(__file__).parent.parent


class TestBuildAndRun(SubProcessChecker):

    @lock(os.path.join(working_path, "test/binary.lock"))
    def setUp(self):
        ''' Compile the tutorial code '''
        self.run_command("make all", working_path, [])

    @pytest.mark.category1
    def test_run_complete_ipu_hardware(self):
        ''' Check that the tutorial code runs on IPU hardware'''

        self.run_command("./tut6 10000 1000 --device ipu",
                         working_path,
                         ["Multiplying matrix of size 10000x1000 by vector of size 1000",
                          "Worst cost seen: 53807",
                          "Multiplication result OK"])

    @pytest.mark.category1
    def test_run_complete_mk1(self):
        ''' Check that the tutorial code runs on Mk1'''

        self.run_command("./tut6 10000 1000 --device model-ipu1",
                         working_path,
                         ["Multiplying matrix of size 10000x1000 by vector of size 1000",
                          "Worst cost seen: 64373",
                          "Multiplication result OK"])

    @pytest.mark.category1
    def test_run_complete_mk2(self):
        ''' Check that the tutorial code runs on Mk2'''

        self.run_command("./tut6 10000 1000 --device model-ipu2",
                         working_path,
                         ["Multiplying matrix of size 10000x1000 by vector of size 1000",
                          "Worst cost seen: 53807",
                          "Multiplication result OK"])
