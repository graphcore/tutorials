from pathlib import Path

import click

from format_converter import py_to_ipynb
from writers import write_nodes_to_json, write_nodes_to_pure_python, autogenerate_path_if_needed


@click.group()
def cli():
    pass


@cli.command()
@click.option('--filename', '-f', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output', '-o', required=True, type=Path,
              help='Absolute or relative path to output jupyter notebook file ')
@click.option('--output_pure_py', '-p', required=False, type=Path,
              help='Absolute or relative path to output pure python file. '
                   'If not provided, will be named automatically.')
def py2ipynb(filename: Path, output: Path, output_pure_py: Path) -> None:
    """
    Command used to generate all outputs with one flow.
    """
    py_text = filename.read_text()
    notebook = py_to_ipynb(py_text)

    write_nodes_to_json(notebook=notebook,
                        output=output)
    write_nodes_to_pure_python(notebook=notebook,
                               output=autogenerate_path_if_needed(filepath=output_pure_py,
                                                                  suffix="_pure_python",
                                                                  template=filename))


if __name__ == '__main__':
    cli()
