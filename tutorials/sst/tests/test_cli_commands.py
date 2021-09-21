import os
from os.path import abspath
from pathlib import Path

import pytest
from click.testing import CliRunner

from sst import py2ipynb
from tests.path_utils import remove_files_if_present, get_tests_dir


STATIC_FILES = Path(get_tests_dir() + os.sep + 'static')

example_input = abspath(STATIC_FILES / "just_py_method.py")
test_output = abspath(STATIC_FILES / "delete_me.json")


@pytest.fixture
def cli_runner_instance():
    return CliRunner()


def test_command_py2ipynb_positive(cli_runner_instance):
    outputs = [test_output]
    remove_files_if_present(outputs)

    result = cli_runner_instance.invoke(py2ipynb, ['--filename', example_input, "--output", test_output])
    print(result.stdout)
    print(result.exception)
    assert result.exit_code == 0
    assert os.path.exists(test_output)

    remove_files_if_present(outputs)


def test_command_py2ipynb_missing_filename(cli_runner_instance):
    result = cli_runner_instance.invoke(py2ipynb, ["--output", test_output])
    assert not os.path.exists(test_output)
    assert result.exit_code == 2


def test_command_py2ipynb_missing_output(cli_runner_instance):
    result = cli_runner_instance.invoke(py2ipynb, ['--filename', example_input])
    assert not os.path.exists(test_output)
    assert result.exit_code == 2
