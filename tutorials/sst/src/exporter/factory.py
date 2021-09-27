from nbconvert import Exporter, MarkdownExporter, NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor

from src.exporter.preprocessors import configure_tag_removal_preprocessor
from src.exporter.pure_python import PurePythonExporter
from src.output_types import OutputTypes


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
