from pathlib import Path

import click

from src.exporters import exporter_factory
from src.format_converter import py_to_ipynb
from src.output_types import OutputTypes, supported_types, EXTENSION2TYPE, TYPE2EXTENSION


@click.command()
@click.option('--source', '-s', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output', '-o', required=True, type=Path,
              help='Absolute or relative path to output file. Output file type is taken from extension. '
                   'Output filepath can be provided without extenstion, then type is taken from --type.')
@click.option('--type', '-f', type=click.Choice(supported_types() + [None]), default=None,
              help='Desired output file type. Parameter is ignored when --output contains specified file extension')
@click.option('--execute/--no-execute', default=True, help='Flag whether the notebook is to be executed or not')
def cli(source: Path, output: Path, type: OutputTypes, execute: bool) -> None:
    """
    Command used to generate all outputs with one flow.
    """
    if output.suffix != '':
        allowed_extensions = list(EXTENSION2TYPE.keys())
        assert output.suffix in allowed_extensions, \
            f'Specified outputy file has type: {output.suffix}, while only {allowed_extensions} are allowed.'
        type = EXTENSION2TYPE[output.suffix]
    elif type is not None:
        output = Path(str(output) + TYPE2EXTENSION[type])
    else:
        raise AttributeError(
            f'Please provide output file type by adding extension to outfile (.md or .ipynb) or specifying that by '
            f'--type parameter [{OutputTypes.MARKDOWN_TYPE}, {OutputTypes.JUPYTER_TYPE}] are allowed.'
        )

    assert output != source, f'Your source file and the expected output file name are the same: {source}, ' \
                             f'specify different outfile name using --output flag.'

    py_text = source.read_text()
    notebook = py_to_ipynb(py_text)

    exporter = exporter_factory(type=type, execute_enabled=execute)
    output_content, _ = exporter.from_notebook_node(notebook)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(output_content)


if __name__ == '__main__':
    cli()  # pragma: no cover
