import os
from os.path import abspath
from pathlib import Path

import pytest
from click.testing import CliRunner

from sst import cli
from tests.path_utils import remove_files_if_present, get_tests_dir


STATIC_FILES = Path(get_tests_dir() + os.sep + 'static')

example_input = abspath(STATIC_FILES / "trivial_mapping_md_code_md.py")
test_output = Path('output_file_name')


@pytest.fixture
def cli_runner_instance():
    return CliRunner()


def test_cli_positive(cli_runner_instance):
    test_output_with_extension = Path(str(test_output) + '.ipynb')
    remove_files_if_present([test_output_with_extension])

    result = cli_runner_instance.invoke(cli, ['--source', example_input, "--output", test_output, "--type", "jupyter"])

    print(result.stdout)
    print(result.exception)

    assert result.exit_code == 0
    assert os.path.exists(test_output_with_extension)

    remove_files_if_present([test_output_with_extension])


def test_cli_missing_filename(cli_runner_instance):
    result = cli_runner_instance.invoke(cli, ["--output", test_output])
    assert not os.path.exists(test_output)
    assert result.exit_code == 2


def test_cli_missing_output(cli_runner_instance):
    result = cli_runner_instance.invoke(cli, ['--source', example_input])
    assert not os.path.exists(test_output)
    assert result.exit_code == 2
