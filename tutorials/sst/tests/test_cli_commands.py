import os
from os.path import abspath
from pathlib import Path

import pytest
from click.testing import CliRunner

from sst import cli
from tests.path_utils import get_tests_dir

STATIC_FILES = Path(get_tests_dir() + os.sep + 'static')

example_input = abspath(STATIC_FILES / "trivial_mapping_md_code_md.py")


@pytest.fixture
def cli_runner_instance():
    return CliRunner()


@pytest.mark.parametrize("type, expected_extension",
                         [('purepython', '.py'), ('markdown', '.md'), ('jupyter', '.ipynb')])
@pytest.mark.parametrize("output_filename", ['filename', 'nested/path/filename'])
def test_cli_positive(cli_runner_instance, tmp_path, type, expected_extension, output_filename):
    outfile_path = tmp_path / output_filename
    expected_output_path = Path(str(outfile_path) + expected_extension)

    result = cli_runner_instance.invoke(cli, ['--source', example_input, "--output", outfile_path, "--type", type])
    print(result.output)
    assert result.exit_code == 0
    assert os.path.exists(expected_output_path)


def test_wrong_path_when_purepython():
    with pytest.raises(AttributeError) as e_info:
        cli_runner_instance.invoke(cli, ['--source', example_input, "--output", example_input, "--type", 'purepython'])


def test_cli_missing_filename(cli_runner_instance):
    result = cli_runner_instance.invoke(cli, ["--output", 'filename'])
    assert result.exit_code == 2


def test_cli_missing_output(cli_runner_instance):
    result = cli_runner_instance.invoke(cli, ['--source', example_input])
    assert result.exit_code == 2
