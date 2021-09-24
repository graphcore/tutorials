from pathlib import Path

from nbconvert import Exporter, MarkdownExporter, NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor

from src.batch_execution import batch_config
from src.format_converter import py_to_ipynb, set_output_extension_and_type
from src.output_types import OutputTypes
from src.output_types import supported_types
from src.preprocessors import configure_tag_removal_preprocessor
from src.pure_python_exporter import PurePythonExporter


def markdown_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    config = configure_tag_removal_preprocessor()

    exporter = MarkdownExporter()
    exporter.register_preprocessor(ExecutePreprocessor(), enabled=execute_enabled)
    exporter.register_preprocessor(TagRemovePreprocessor(config=config))

    return exporter


def notebook_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    exporter = NotebookExporter()
    exporter.register_preprocessor(ExecutePreprocessor(), enabled=execute_enabled)

    return exporter


def pure_python_exporter(execute_enabled: bool) -> Exporter:
    return PurePythonExporter()


TYPE2EXPORTER = {
    OutputTypes.JUPYTER_TYPE: notebook_exporter_with_preprocessors,
    OutputTypes.MARKDOWN_TYPE: markdown_exporter_with_preprocessors,
    OutputTypes.PUREPYTHON_TYPE: pure_python_exporter
}


def exporter_factory(type: OutputTypes, execute_enabled: bool) -> Exporter:
    exporter_factory = TYPE2EXPORTER.get(type)
    exporter = exporter_factory(execute_enabled=execute_enabled)
    return exporter


def execute_single_exporter(source: Path, output: Path, output_type: OutputTypes, execute: bool):
    py_text = source.read_text()
    notebook = py_to_ipynb(py_text)

    exporter = exporter_factory(type=output_type, execute_enabled=execute)
    output_content, _ = exporter.from_notebook_node(notebook)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(output_content)


def execute_multiple_exporters(input_directory: Path, output_directory: Path, config_path: Path, execute: bool):
    tutorial_configs = batch_config(config_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    for tc in tutorial_configs:
        for supported_type in supported_types():
            output, output_type = set_output_extension_and_type(output_directory / tc.name, supported_type)
            execute_single_exporter(
                execute=execute,
                output=output,
                source=input_directory / tc.source,
                output_type=output_type
            )
