# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import json
from pathlib import Path

import pytest
from deepdiff import DeepDiff
from nbformat import NotebookNode

from src.format_converter import py_to_ipynb
from tests.test_utils.path import get_unit_test_static_files_dir

STATIC_FILES = get_unit_test_static_files_dir()


@pytest.mark.parametrize(
    "file_name",
    ['trivial_mapping_md_code_md', 'just_py_method', 'two_markdowns', 'comments_before_markdown'],
)
def test_py2json(file_name):
    python_file_path, json_file_path = Path(file_name + '.py'), Path(file_name + '.json')

    file_py = (STATIC_FILES / python_file_path).read_text()
    transformed_py = py_to_ipynb(file_py)

    file_json = (STATIC_FILES / json_file_path).read_text()
    json_dict = json.loads(file_json)

    diff = DeepDiff(json_dict['cells'], transformed_py['cells'], ignore_type_in_groups=[(NotebookNode, dict)],
                    exclude_regex_paths=r"root\[\d+\]\['id'\]")

    assert diff == {}
