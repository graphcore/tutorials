# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from pathlib import Path

from tqdm import tqdm
from nbconvert.exporters.exporter import ResourcesDict
from nbconvert.writers import FilesWriter

from src.batch_execution import batch_config
from src.constants import NBCONVERT_RESOURCE_OUTPUT_EXT_KEY, IMAGES_DIR, NBCONVERT_RESOURCE_OUTPUT_DIR_KEY
from src.exporter.factory import exporter_factory
from src.format_converter import py_to_ipynb
from src.output_types import supported_types, OutputTypes
from src.utils.file import set_output_extension_and_type, output_path_code


def execute_conversion(source: Path, output: Path, output_type: OutputTypes, execute: bool):
    py_text = source.read_text()
    notebook = py_to_ipynb(py_text)

    exporter = exporter_factory(type=output_type, execute_enabled=execute)

    output_content, resources = exporter.from_notebook_node(
        notebook,
        resources={
            NBCONVERT_RESOURCE_OUTPUT_DIR_KEY: output.parent / f"{source.stem}_{IMAGES_DIR}"
        }
    )

    save_conversion_results(output, output_content, resources)


def save_conversion_results(output: Path, output_content: str, resources: ResourcesDict):
    output.parent.mkdir(parents=True, exist_ok=True)

    if resources and NBCONVERT_RESOURCE_OUTPUT_EXT_KEY in resources:
        del resources[NBCONVERT_RESOURCE_OUTPUT_EXT_KEY]

    writer = FilesWriter(build_directory=str(output.parent))
    writer.write(output=output_content, resources=resources, notebook_name=str(output.name))


def execute_multiple_conversions(source_directory: Path, output_directory: Path, config_path: Path, execute: bool):
    conversion_configs = batch_config(config_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    for tc in tqdm(conversion_configs, desc="SST All Configs", leave=True):
        for supported_type in tqdm(supported_types(), desc="SST Config", leave=False):
            output, output_type = set_output_extension_and_type(output_directory / tc.name, supported_type)

            output = output_path_code(output) if output_type == OutputTypes.CODE_TYPE else output

            execute_conversion(
                execute=execute if supported_type is OutputTypes.MARKDOWN_TYPE else False,
                output=output,
                source=source_directory / tc.source,
                output_type=output_type
            )
