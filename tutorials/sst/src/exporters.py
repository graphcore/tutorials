from nbconvert import Exporter, MarkdownExporter, NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor
from traitlets.config import Config

from src.constants import REMOVE_OUTPUT_TAG
from src.output_types import OutputTypes
from src.python_exporter import PythonExporter


def markdown_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    c = Config()
    c.ExecutePreprocessor.enabled = execute_enabled
    c.TagRemovePreprocessor.remove_all_outputs_tags = (REMOVE_OUTPUT_TAG,)
    c.TagRemovePreprocessor.enabled = True

    exporter = MarkdownExporter()
    exporter.register_preprocessor(ExecutePreprocessor(config=c))
    exporter.register_preprocessor(TagRemovePreprocessor(config=c))

    return exporter


def notebook_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    exporter = NotebookExporter()
    exporter.register_preprocessor(ExecutePreprocessor(), enabled=execute_enabled)

    return exporter


def pure_python_exporter(execute_enabled: bool) -> Exporter:
    return PythonExporter()


TYPE2EXPORTER = {
    OutputTypes.JUPYTER_TYPE: notebook_exporter_with_preprocessors,
    OutputTypes.MARKDOWN_TYPE: markdown_exporter_with_preprocessors,
    OutputTypes.PUREPYTHON_TYPE: pure_python_exporter
}


def exporter_factory(type: OutputTypes, execute_enabled: bool) -> Exporter:
    exporter_factory = TYPE2EXPORTER.get(type)
    exporter = exporter_factory(execute_enabled=execute_enabled)
    return exporter
