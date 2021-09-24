from pathlib import Path

from src.batch_execution import batch_config
from src.exporter.factory import exporter_factory
from src.format_converter import py_to_ipynb
from src.output_types import supported_types, OutputTypes
from src.utils.file import set_output_extension_and_type


def execute_conversion(source: Path, output: Path, output_type: OutputTypes, execute: bool):
    py_text = source.read_text()
    notebook = py_to_ipynb(py_text)

    exporter = exporter_factory(type=output_type, execute_enabled=execute)
    output_content, _ = exporter.from_notebook_node(notebook)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(output_content)


def execute_multiple_conversions(source_directory: Path, output_directory: Path, config_path: Path, execute: bool):
    tutorial_configs = batch_config(config_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    for tc in tutorial_configs:
        for supported_type in supported_types():
            output, output_type = set_output_extension_and_type(output_directory / tc.name, supported_type)
            execute_conversion(
                execute=execute,
                output=output,
                source=source_directory / tc.source,
                output_type=output_type
            )
