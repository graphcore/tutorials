# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os

import pytest
from click.testing import CliRunner

from src.utils.click import print_exception
from tests.test_utils.logger_utils import disable_logging
from tests.test_utils.path import get_unit_test_static_files_dir
from sst import cli

STATIC_FILES = get_unit_test_static_files_dir()


@pytest.fixture
def cli_runner_instance():
    disable_logging()
    return CliRunner()


def test_cli_batch_convert(cli_runner_instance, tmp_path):
    example_config = STATIC_FILES / "conversion_config.yml"
    output_dir = tmp_path / 'output_dir'
    source_dir = STATIC_FILES.parent.parent

    result = cli_runner_instance.invoke(cli, ['batch-convert',
                                              '--config', example_config, "--output-dir", output_dir, "--source-dir",
                                              source_dir, "--no-execute"
                                              ])

    print_exception(result)

    assert result.exit_code == 0

    assert len(os.listdir(output_dir)) == 6


@pytest.mark.parametrize("config_path", ['conversion_config_incorrect.yml', 'conversion_config_empty.yml'])
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
