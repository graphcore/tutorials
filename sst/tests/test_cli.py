# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
from click.testing import CliRunner

from src.utils.click import print_exception
from sst import cli


@pytest.fixture
def cli_runner_instance():
    return CliRunner()


@pytest.mark.parametrize("command", ['convert', 'convert2all', 'batch-convert'])
def test_cli_when_help_command(cli_runner_instance, command):
    result = cli_runner_instance.invoke(cli, [command, '--help'])
    print_exception(result)
    assert result.exit_code == 0


def test_cli_when_help_no_command(cli_runner_instance):
    result = cli_runner_instance.invoke(cli, ['--help'])
    print_exception(result)
    assert result.exit_code == 0
