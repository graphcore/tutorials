# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from nbconvert import Exporter, NotebookExporter, MarkdownExporter
from nbconvert.preprocessors import TagRemovePreprocessor, ExtractOutputPreprocessor
from traitlets.config import Config

from src.exporter.code_exporter import CodeExporter
from src.exporter.execute_preprocessor_with_progress_bar import ExecutePreprocessorWithProgressBar
from src.exporter.preprocessors import configure_tag_removal_preprocessor, \
    configure_extract_outputs_preprocessor, \
    configure_copyright_regex_removal_preprocessor, \
    RegexWithFlagsRemovePreprocessor
from src.output_types import OutputTypes


def markdown_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    exporter = MarkdownExporter()
    exporter.register_preprocessor(ExecutePreprocessorWithProgressBar(), enabled=execute_enabled)

    config = Config()
    for apply_configuration in [
        configure_tag_removal_preprocessor,
        configure_copyright_regex_removal_preprocessor,
        configure_extract_outputs_preprocessor
    ]:
        config = apply_configuration(config)

    for preprocessor in [
        TagRemovePreprocessor(config=config),
        RegexWithFlagsRemovePreprocessor(config=config),
        ExtractOutputPreprocessor(config=config)
    ]:
        exporter.register_preprocessor(preprocessor, enabled=True)

    return exporter


def notebook_exporter_with_preprocessors(execute_enabled: bool) -> Exporter:
    exporter = NotebookExporter()
    exporter.register_preprocessor(ExecutePreprocessorWithProgressBar(), enabled=execute_enabled)

    return exporter


def code_exporter(execute_enabled: bool) -> Exporter:
    return CodeExporter()


TYPE2EXPORTER = {
    OutputTypes.JUPYTER_TYPE: notebook_exporter_with_preprocessors,
    OutputTypes.MARKDOWN_TYPE: markdown_exporter_with_preprocessors,
    OutputTypes.CODE_TYPE: code_exporter
}


def exporter_factory(type: OutputTypes, execute_enabled: bool) -> Exporter:
    exporter_factory = TYPE2EXPORTER.get(type)
    exporter = exporter_factory(execute_enabled=execute_enabled)
    return exporter
