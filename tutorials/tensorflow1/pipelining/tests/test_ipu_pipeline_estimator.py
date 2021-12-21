# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from tutorials_tests.testing_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestIpuPipelineEstimator(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_run_ipu_pipeline_estimator(self):
        """ Check answers/ipu_pipeline_estimator.py works """
        self.run_command("python answers/ipu_pipeline_estimator.py",
                         working_path,
                         "Program ran successfully")
