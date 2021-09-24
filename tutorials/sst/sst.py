from pathlib import Path
from typing import Tuple

import click
from tqdm import tqdm

from src.exporters import exporter_factory
from src.format_converter import py_to_ipynb
from src.output_types import OutputTypes, supported_types, EXTENSION2TYPE, TYPE2EXTENSION


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source', '-s', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output', '-o', required=True, type=Path,
              help='Absolute or relative path to output file. Output file type is taken from extension. '
                   'Output filepath can be provided without extenstion, then type is taken from --type.')
@click.option('--type', '-t', type=click.Choice(supported_types()), default=None,
              help='Desired output file type. Parameter is ignored when --output contains specified file extension')
@click.option('--execute/--no-execute', default=True, help='Flag whether the file is to be executed or not')
def convert(source: Path, output: Path, type: OutputTypes, execute: bool) -> None:
    """
    Command used to generate all outputs with one flow.
    """
    output, type = set_output_extension_and_type(output, type)

    assert source.suffix == '.py', 'Only python file can be single source file'
    assert output != source, f'Your source file and the expected output file name are the same: {source}, ' \
                             f'specify different outfile name using --output flag.'

    transform_python_file(source, output, type, execute)


@cli.command()
@click.option('--source', '-s', required=True, type=Path,
              help='Absolute or relative path to python file to be converted')
@click.option('--output-dir', '-o', required=False, type=Path, help='Absolute or relative path to output directory.')
def convert2all(source: Path, output_dir: Path):
    assert source.suffix == '.py', 'Only python file can be single source file'
    if output_dir is None:
        output_dir = source.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = output_dir / source.stem

    configuration = [
        [output_filename.with_suffix('.md'), OutputTypes.MARKDOWN_TYPE, True],
        [output_filename.with_stem(source.stem + '_pure').with_suffix('.py'), OutputTypes.PUREPYTHON_TYPE, False],
        [output_filename.with_suffix('.ipynb'), OutputTypes.JUPYTER_TYPE, False]
    ]

    for outfile, output_type, execution in tqdm(configuration):
        transform_python_file(source=source, output=outfile, type=output_type, execute=execution)


def set_output_extension_and_type(output: Path, type: OutputTypes) -> Tuple[Path, OutputTypes]:
    """
    If output without extension but specified type -> add extension to output
    If output with extension -> overwrite current type
    If output with extension but not allowed extension -> raise AssertionError
    If output without extension and type is None -> raise AttributeError
    """
    if output.suffix != '':
        allowed_extensions = list(EXTENSION2TYPE.keys())
        assert output.suffix in allowed_extensions, \
            f'Specified outputy file has type: {output.suffix}, while only {allowed_extensions} are allowed.'
        type = EXTENSION2TYPE[output.suffix]
    elif type is not None:
        output = output.with_suffix(TYPE2EXTENSION[type])
        print(output)
    else:
        raise AttributeError(
            f'Please provide output file type by adding extension to outfile (.md or .ipynb) or specifying that by '
            f'--type parameter [{OutputTypes.MARKDOWN_TYPE}, {OutputTypes.JUPYTER_TYPE}] are allowed.'
        )

    return output, type


def transform_python_file(source: Path, output: Path, type: OutputTypes, execute: bool) -> None:
    py_text = source.read_text()
    notebook = py_to_ipynb(py_text)

    exporter = exporter_factory(type=type, execute_enabled=execute)
    output_content, _ = exporter.from_notebook_node(notebook)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(output_content)


if __name__ == '__main__':
    cli()  # pragma: no cover
