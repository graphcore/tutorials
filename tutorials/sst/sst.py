from pathlib import Path

import click
from nbformat import v4

from format_converter import py_to_ipynb


@click.group()
def cli():
    pass


@cli.command()
@click.option('--filename', '-f', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output', '-o', required=True, type=Path,
              help='Absolute or relative path to output jupyter notebook file ')
def py2ipynb(filename: Path, output: Path) -> None:
    """
    Description
    """
    py_text = filename.read_text()
    notebook = py_to_ipynb(py_text)

    jsonform = v4.writes(notebook)
    with open(output, "w") as fpout:
        fpout.write(jsonform)


if __name__ == '__main__':
    cli()
