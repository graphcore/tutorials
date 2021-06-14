# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestComplete(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_model_fp16(self):
        # Check whether the model compiles and executes in FP16
        self.run_command("python3 walkthrough.py --model-half",
                         working_path,
                         "Eval accuracy on IPU")

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_model_data_fp16(self):
        # Check whether the model compiles and executes in FP16
        self.run_command("python3 walkthrough.py --model-half --data-half",
                         working_path,
                         "Eval accuracy on IPU")

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_model_data_accum_fp16(self):
        # Check whether the model compiles and executes in FP16
        self.run_command("python3 walkthrough.py --model-half --data-half --optimizer-half",
                         working_path,
                         "Eval accuracy on IPU")

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_stochastic_rounding(self):
        # Check whether the model compiles and executes in FP16
        self.run_command("python3 walkthrough.py --model-half --data-half --optimizer-half --stochastic-rounding",
                         working_path,
                         "Eval accuracy on IPU")

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_all_fp16(self):
        # Check whether the model compiles and executes in FP16
        self.run_command("python3 walkthrough.py --model-half --data-half --optimizer-half --stochastic-rounding --partials-half",
                         working_path,
                         "Eval accuracy on IPU")
