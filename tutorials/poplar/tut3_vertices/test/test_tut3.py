# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tutorials_tests.testing_util import SubProcessChecker
from tutorials_tests.xdist_util import lock

working_path = Path(__file__).parent


class TestBuildAndRun(SubProcessChecker):

    @lock(os.path.join(working_path, "binary.lock"))
    def setUp(self):
        ''' Compile the complete version of the tutorial code '''
        self.run_command("make all", working_path, [])

    @pytest.mark.category1
    def test_run_complete(self):
        ''' Check that the complete version of the tutorial code runs '''

        self.run_command("../complete/tut3_complete",
                         working_path.parent.joinpath("complete"),
                         ["Program complete",
                          "v2: {7,6,4.5,2.5}"])
