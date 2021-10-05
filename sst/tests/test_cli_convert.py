# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import re

import pytest
from click.testing import CliRunner

from src.constants import SST_HIDE_OUTPUT_TAG, COPYRIGHT_TAG
from src.utils.click import print_exception
from sst import cli
from tests.test_utils.path import get_unit_test_static_files_dir

STATIC_FILES = get_unit_test_static_files_dir()
TRIVIAL_MAPPING_SOURCE_PATH = STATIC_FILES / "trivial_mapping_md_code_md.py"


@pytest.fixture
def cli_runner_instance():
    return CliRunner()


@pytest.mark.parametrize("type, expected_extension",
                         [('code', '.py'), ('markdown', '.md'), ('jupyter', '.ipynb')])
@pytest.mark.parametrize("output_filename", ['filename', 'nested/path/filename'])
def test_cli_positive(cli_runner_instance, tmp_path, type, expected_extension, output_filename):
    outfile_path = tmp_path / output_filename
    expected_output_path = outfile_path.with_suffix(expected_extension)
    result = cli_runner_instance.invoke(cli, [
        'convert', '--source', TRIVIAL_MAPPING_SOURCE_PATH, "--output", outfile_path, "--type", type
    ])

    print_exception(result)

    assert result.exit_code == 0
    assert os.path.exists(expected_output_path)


@pytest.mark.parametrize("output_filename", ['file.py', 'nested/file.md', 'file.ipynb'])
def test_cli_positive_when_no_type(cli_runner_instance, tmp_path, output_filename):
    outfile_path = tmp_path / output_filename

    result = cli_runner_instance.invoke(cli,
                                        ['convert', '--source', TRIVIAL_MAPPING_SOURCE_PATH, "--output", outfile_path])

    print_exception(result)

    assert result.exit_code == 0
    assert os.path.exists(outfile_path)


@pytest.mark.parametrize("output_filename", ['file.txt', 'nested/file.avi'])
def test_cli_when_wrong_extension(cli_runner_instance, tmp_path, output_filename):
    outfile_path = tmp_path / output_filename
    with pytest.raises(AssertionError):
        result = cli_runner_instance.invoke(cli, ['convert', '--source', TRIVIAL_MAPPING_SOURCE_PATH, "--output",
                                                  outfile_path])
        if result.exception:
            raise result.exception


def test_cli_when_missing_output_extension_or_type(cli_runner_instance):
    with pytest.raises(AttributeError):
        result = cli_runner_instance.invoke(cli,
                                            ['convert', '--source', TRIVIAL_MAPPING_SOURCE_PATH, '--output', 'file'])
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

    print_exception(result)

    assert result.exit_code == 0

    generated_markdown = outfile_path.read_text()
    expected_markdown = expected_markdown_path.read_text()

    assert generated_markdown == expected_markdown


def test_wrong_path_when_code_export(cli_runner_instance):
    with pytest.raises(AssertionError) as e_info:
        result = cli_runner_instance.invoke(cli, [
            'convert', '--source', TRIVIAL_MAPPING_SOURCE_PATH, "--output", TRIVIAL_MAPPING_SOURCE_PATH, "--type",
            'code'
        ])
        if result.exception:
            raise result.exception


def test_cli_positive_markdown_output_removal_by_tags(cli_runner_instance, tmp_path):
    example_input = STATIC_FILES / "code_blocks_with_outputs_to_be_removed.py"
    outfile = tmp_path / 'output'
    outfile_path = tmp_path / 'output.md'

    result = cli_runner_instance.invoke(cli, [
        'convert', '--source', example_input, "--output", outfile, "--type", "markdown", "--execute"
    ])

    print_exception(result)

    assert result.exit_code == 0

    # In this file  we expected exactly one print statement to work
    with open(outfile_path) as f:
        actual_contents = f.read()
        assert "Hello sunshine1!" in actual_contents
        assert "Goodbye sunshine2!" not in actual_contents
        assert "Hello sunshine3!" not in actual_contents
        assert "Goodbye sunshine4!" not in actual_contents
        assert SST_HIDE_OUTPUT_TAG not in actual_contents


def test_cli_positive_markdown_output_removal_by_regex_copyright(cli_runner_instance, tmp_path):
    example_input = STATIC_FILES / "copyright_removal.py"
    outfile = tmp_path / 'output'
    outfile_path = tmp_path / 'output.md'

    result = cli_runner_instance.invoke(cli, [
        'convert', '--source', example_input, "--output", outfile, "--type", "markdown", "--execute"
    ])

    print_exception(result)
    assert result.exit_code == 0

    markdown_content = outfile_path.read_text()
    assert "I am the markdown!" in markdown_content

    copyright_occurrences = len(list(re.finditer(COPYRIGHT_TAG, markdown_content.lower())))
    assert copyright_occurrences == 0


def test_cli_positive_code_only_output_removal_by_regex_copyright(cli_runner_instance, tmp_path):
    example_input = STATIC_FILES / "copyright_removal.py"
    outfile = tmp_path / 'output'
    outfile_path = tmp_path / 'output.py'

    result = cli_runner_instance.invoke(cli, [
        'convert', '--source', example_input, "--output", outfile, "--type", "code", "--execute"
    ])

    print_exception(result)
    assert result.exit_code == 0

    script_content = outfile_path.read_text()
    assert "I am the only markdown cell around here!" not in script_content
    assert "# copyright" in script_content


def test_cli_positive_markdown_output_extraction(cli_runner_instance, tmp_path):
    example_input = STATIC_FILES / "output_extraction.py"
    output_filename = "output.md"
    outfile = tmp_path / 'output'
    outfile_path = tmp_path / output_filename

    result = cli_runner_instance.invoke(cli, [
        'convert', '--source', example_input, "--output", outfile, "--type", "markdown", "--execute"
    ])

    print_exception(result)

    assert result.exit_code == 0

    files = os.listdir(outfile_path.parent)
    assert len(files) == 2
    assert output_filename in files

    extracted_outputs = os.listdir(outfile_path.parent / "output-outputs")
    assert "output_1_0.png" in extracted_outputs


def test_cli_missing_filename(cli_runner_instance):
    result = cli_runner_instance.invoke(cli, ['convert', "--output", 'filename'])
    assert result.exit_code == 2


def test_cli_missing_output(cli_runner_instance):
    result = cli_runner_instance.invoke(cli, ['convert', '--source', TRIVIAL_MAPPING_SOURCE_PATH])
    assert result.exit_code == 2


def test_cli_convert_when_input_file_is_not_py(cli_runner_instance):
    with pytest.raises(AssertionError) as e_info:
        result = cli_runner_instance.invoke(cli, ['convert', '--source', 'input.txt', "--output", 'output.py'])
        if result.exception:
            raise result.exception
