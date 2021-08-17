# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path
import pytest
import tempfile
from shutil import copy

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker
from examples_tests.xdist_util import lock

working_path = Path(__file__).parent


class TestBuildAndRun(SubProcessChecker):

    @lock(os.path.join(working_path, "binary.lock"))
    def setUp(self):
        ''' Compile the start here and complete versions of the tutorial code '''
        self.run_command("make all", working_path, [])

    @pytest.mark.category1
    def test_run_ipu_hardware(self):
        ''' Check that the hardware version of the tutorial code runs '''

        self.run_command("./tut4_ipu_hardware",
                         working_path,
                         ["Program complete",
                          "Memory Usage:"])

    @pytest.mark.category1
    def test_run_ipu_model(self):
        ''' Check that the IPUModel version of the tutorial code runs '''

        self.run_command("./tut4_ipu_model",
                         working_path,
                         ["Program complete",
                          "Memory Usage:"])

    @pytest.mark.category1
    def test_run_cpp_example(self):
        ''' Check that the CppExample can open a profile.pop report '''
        # Set environment var to collect reports
        env = os.environ.copy()
        env['POPLAR_ENGINE_OPTIONS'] = '{"autoReport.all": "true"}'

        with tempfile.TemporaryDirectory() as temporary_path:
            # Copy binaries to temp directory
            copy(os.path.dirname(os.path.abspath(__file__)) + "/tut4_ipu_model",
                 os.path.join(temporary_path, "tut4_ipu_model"))
            copy(os.path.dirname(os.path.abspath(__file__)) + "/CppExample", os.path.join(temporary_path, "CppExample"))

            # Execute ipu_model that will collect reports
            self.run_command("./tut4_ipu_model",
                             temporary_path,
                             ["Program complete",
                              "Memory Usage:"],
                             env=env)
            # Inspect reports
            self.run_command("./CppExample",
                             temporary_path,
                             ["Example information from profile"])
