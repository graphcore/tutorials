# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pathlib
import sys
import pytest

# testing_util import relies on pytest.ini at the root
# of the repository
import tutorials_tests.testing_util as testing_util


@pytest.mark.category1
@pytest.mark.ipus(2)
def test_pipelining_convergence():
    """Run script "pipelining.py" with default settings, failures
    are reported explicitly and check that the loss has converged."""

    # Run the file called pipelining.py in the same folder as this test file.
    working_directory = pathlib.Path(__file__).absolute().parent
    out = testing_util.run_command_fail_explicitly(
        [sys.executable, "pipelining.py"], working_directory
    )

    # Get the final loss from the output.
    loss_regex = r"loss: ([\d.]+)"
    result = testing_util.parse_results_with_regex(out, loss_regex)

    # Check the result against expected values
    assert (
        len(result) > 0 and len(result[0]) > 0
    ), "Results are empty loss could not be parsed"
    loss = result[0][-1]
    assert loss > 0.001, f"Loss was lower than expected: {loss}"
    assert loss < 0.02, f"Loss was higher than expected: {loss}"
