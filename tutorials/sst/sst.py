from pathlib import Path

import click
from nbconvert import NotebookExporter, MarkdownExporter

from format_converter import py2ipynb


@click.command()
@click.option('--source', '-s', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output', '-o', required=True, type=Path,
              help='Absolute or relative path to output file without extension')
@click.option('--type', '-f', type=click.Choice(['jupyter', 'markdown']), help='Desired output file type')
@click.option('--execute/--no-execute', default=True, help='Flag whether the notebook is to be executed or not')
def cli(source: Path, output: Path, type: str, execute: bool) -> None:
    """
    Description
    """
    py_text = source.read_text()
    notebook = py2ipynb(py_text)

    exporter = NotebookExporter() if type == 'jupyter' else MarkdownExporter()
    if execute:
        exporter.register_preprocessor('nbconvert.preprocessors.ExecutePreprocessor', enabled=True)

    output_content, _ = exporter.from_notebook_node(notebook)
    filename = str(output) + exporter.file_extension

    with open(filename, "w") as fpout:
        fpout.write(output_content)


if __name__ == '__main__':
    cli()
