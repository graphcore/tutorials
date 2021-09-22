from pathlib import Path

import click
from nbconvert import NotebookExporter

from src.constants import EXECUTE_PREPROCESSOR, TYPE2EXPORTER
from src.format_converter import py_to_ipynb


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
@click.option('--type', '-f', type=click.Choice(['jupyter', 'markdown', 'purepython']), help='Desired output file type')
@click.option('--execute/--no-execute', default=True, help='Flag whether the notebook is to be executed or not')
def cli(source: Path, output: Path, type: str, execute: bool) -> None:
    """
    Command used to generate all outputs with one flow.
    """
    py_text = source.read_text()
    notebook = py_to_ipynb(py_text)

    exporter_class = TYPE2EXPORTER.get(type, NotebookExporter)
    exporter = exporter_class()
    if execute:
        exporter.register_preprocessor(EXECUTE_PREPROCESSOR, enabled=True)

    output_content, _ = exporter.from_notebook_node(notebook)

    filename = construct_output_filename(outputname=output, extension=exporter.file_extension, input_name=source)

    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(output_content)


if __name__ == '__main__':
    cli()  # pragma: no cover
