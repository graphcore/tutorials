from pathlib import Path

import click
from nbconvert import Exporter

from src.exporters import exporter_factory
from src.format_converter import py_to_ipynb
from src.output_types import OutputTypes, supported_types


def construct_output_filename(outputname: Path, extension: str, input_name: Path) -> Path:
    filename = str(outputname) + extension
    assert not filename == str(input_name), f'Your source file and the expected output file name are the same: ' \
                                            f'{input_name}, specify different outfile name using --output flag.'
    return Path(filename)


@click.command()
@click.option('--source', '-s', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output', '-o', required=True, type=Path,
              help='Absolute or relative path to output file without extension')
@click.option('--type', '-f', type=click.Choice(supported_types()), help='Desired output file type')
@click.option('--execute/--no-execute', default=True, help='Flag whether the notebook is to be executed or not')
def cli(source: Path, output: Path, type: OutputTypes, execute: bool) -> None:
    """
    Command used to generate all outputs with one flow.
    """
    py_text = source.read_text()
    notebook = py_to_ipynb(py_text)

    exporter = exporter_factory(type=type, execute_enabled=execute)
    output_content, _ = exporter.from_notebook_node(notebook)

    filename = construct_output_filename(outputname=output, extension=exporter.file_extension, input_name=source)

    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(output_content)


def create_filename(exporter: Exporter, output: Path, source: Path) -> Path:
    filename = str(output) + exporter.file_extension
    assert not filename == str(source), f'Your source file and the expected output file name are the same: {source}, ' \
                                        f'specify different outfile name using --output flag.'
    return Path(filename)


if __name__ == '__main__':
    cli()  # pragma: no cover
