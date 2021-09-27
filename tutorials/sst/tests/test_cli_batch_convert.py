import os

import pytest
from click.testing import CliRunner

from src.utils.click import print_exception
from src.utils.path import STATIC_FILES
from sst import cli


@pytest.fixture
def cli_runner_instance():
    return CliRunner()


def test_cli_batch_convert(cli_runner_instance, tmp_path):
    example_config = STATIC_FILES / "tutorial_config.yml"
    output_dir = tmp_path / 'output_dir'
    source_dir = STATIC_FILES.parent.parent

    result = cli_runner_instance.invoke(cli, ['batch-convert',
                                              '--config', example_config, "--output-dir", output_dir, "--source-dir",
                                              source_dir, "--no-execute"
                                              ])

    print_exception(result)

    assert result.exit_code == 0

    assert len(os.listdir(output_dir)) == 6


@pytest.mark.parametrize("config_path", ['tutorial_config_incorrect.yml', 'tutorial_config_empty.yml'])
def test_cli_batch_convert_incorrect_config(cli_runner_instance, tmp_path, config_path):
    example_config = STATIC_FILES / config_path
    output_dir = tmp_path / 'output_dir'
    source_dir = STATIC_FILES.parent.parent

    with pytest.raises(AttributeError) as e_info:
        result = cli_runner_instance.invoke(cli, [
            'batch-convert',
            '--config', example_config,
            "--output-dir", output_dir,
            "--source-dir", source_dir,
            "--no-execute"
        ])

        print_exception(result)

        if result.exception:
            raise result.exception
