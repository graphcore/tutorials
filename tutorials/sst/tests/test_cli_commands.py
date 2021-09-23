import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from sst import cli
from tests.path_utils import get_tests_dir

STATIC_FILES = Path(get_tests_dir() + os.sep + 'static')

example_input = STATIC_FILES / "trivial_mapping_md_code_md.py"


@pytest.fixture
def cli_runner_instance():
    return CliRunner()


@pytest.mark.parametrize("type, expected_extension",
                         [('purepython', '.py'), ('markdown', '.md'), ('jupyter', '.ipynb')])
@pytest.mark.parametrize("output_filename", ['filename', 'nested/path/filename'])
def test_cli_positive(cli_runner_instance, tmp_path, type, expected_extension, output_filename):
    outfile_path = tmp_path / output_filename
    expected_output_path = Path(str(outfile_path) + expected_extension)
    result = cli_runner_instance.invoke(cli, ['convert', '--source', example_input, "--output", outfile_path, "--type",
                                              type])

    if result.exception:
        print(result.exception)

    assert result.exit_code == 0
    assert os.path.exists(expected_output_path)


@pytest.mark.parametrize("output_filename", ['file.py', 'nested/file.md', 'file.ipynb'])
def test_cli_positive_when_no_type(cli_runner_instance, tmp_path, output_filename):
    outfile_path = tmp_path / output_filename

    result = cli_runner_instance.invoke(cli, ['convert', '--source', example_input, "--output", outfile_path])

    if result.exception:
        print(result.exception)

    assert result.exit_code == 0
    assert os.path.exists(outfile_path)


@pytest.mark.parametrize("output_filename", ['file.txt', 'nested/file.avi'])
def test_cli_when_wrong_extension(cli_runner_instance, tmp_path, output_filename):
    outfile_path = tmp_path / output_filename
    with pytest.raises(AssertionError):
        result = cli_runner_instance.invoke(cli, ['convert', '--source', example_input, "--output", outfile_path])
        if result.exception:
            raise result.exception


def test_cli_when_missing_output_extension_or_type(cli_runner_instance):
    with pytest.raises(AttributeError):
        result = cli_runner_instance.invoke(cli, ['convert', '--source', example_input, '--output', 'file'])
        if result.exception:
            raise result.exception


def test_py_file_with_import(cli_runner_instance, tmp_path):
    file_path = STATIC_FILES / 'py_with_import.py'
    expected_markdown_path = STATIC_FILES / 'py_with_import.md'

    outfile = tmp_path / 'output'
    outfile_path = tmp_path / 'output.md'
    result = cli_runner_instance.invoke(cli, [
        'convert', '--source', file_path, "--output", outfile, "--type", "markdown", "--execute"
    ])

    if result.exception:
        print(result.exception)

    assert result.exit_code == 0

    generated_markdown = outfile_path.read_text()
    expected_markdown = expected_markdown_path.read_text()

    assert generated_markdown == expected_markdown


def test_wrong_path_when_purepython():
    with pytest.raises(AttributeError) as e_info:
        cli_runner_instance.invoke(cli, [
            'convert', '--source', example_input, "--output", example_input, "--type", 'purepython'
        ])


def test_cli_positive_markdown_output_removal_by_tags(cli_runner_instance, tmp_path):
    example_input = STATIC_FILES / "code_blocks_with_outputs_to_be_removed.py"
    outfile = tmp_path / 'output'
    outfile_path = tmp_path / 'output.md'

    result = cli_runner_instance.invoke(cli, [
        'convert', '--source', example_input, "--output", outfile, "--type", "markdown", "--execute"
    ])

    if result.exception:
        print(result.exception)

    assert result.exit_code == 0

    # In this file  we expected exactly one print statement to work
    with open(outfile_path) as f:
        actual_contents = f.read()
        assert "Hello sunshine1!" in actual_contents
        assert "Goodbye sunshine2!" not in actual_contents
        assert "Hello sunshine3!" not in actual_contents
        assert "Goodbye sunshine4!" not in actual_contents


def test_cli_missing_filename(cli_runner_instance):
    result = cli_runner_instance.invoke(cli, ['convert', "--output", 'filename'])
    assert result.exit_code == 2


def test_cli_missing_output(cli_runner_instance):
    result = cli_runner_instance.invoke(cli, ['convert', '--source', example_input])
    assert result.exit_code == 2


def test_cli_convert_when_input_file_is_not_py(cli_runner_instance):
    with pytest.raises(AssertionError) as e_info:
        result = cli_runner_instance.invoke(cli, ['convert', '--source', 'input.txt', "--output", 'output.py'])
        if result.exception:
            raise result.exception


def test_cli_convert2all_when_input_file_is_not_py(cli_runner_instance):
    with pytest.raises(AssertionError) as e_info:
        result = cli_runner_instance.invoke(cli, ['convert2all', '--source', 'input.txt'])
        if result.exception:
            raise result.exception


def test_cli_convert2all_when_correct_input(cli_runner_instance, tmp_path):
    result = cli_runner_instance.invoke(cli, ['convert2all', '--source', example_input, '--output-dir', tmp_path])
    assert result.exit_code == 0
