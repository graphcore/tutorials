from pathlib import Path

import click
from nbconvert import NotebookExporter, MarkdownExporter

from format_converter import py_to_ipynb
from python_exporter import PythonExporter


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

    typ2exporters = {
        'jupyter': NotebookExporter,
        'markdown': MarkdownExporter,
        'purepython': PythonExporter
    }

    exporter = typ2exporters[type]()
    if execute:
        exporter.register_preprocessor('nbconvert.preprocessors.ExecutePreprocessor', enabled=True)

    output_content, _ = exporter.from_notebook_node(notebook)
    filename = str(output) + exporter.file_extension

    assert not filename == str(source), 'Do not overwrite your python file!'

    with open(filename, "w") as fpout:
        fpout.write(output_content)


if __name__ == '__main__':
    cli()
