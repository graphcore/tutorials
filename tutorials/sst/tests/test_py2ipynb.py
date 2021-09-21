import json
from pathlib import Path

import pytest
from deepdiff import DeepDiff
from nbformat import NotebookNode

from format_converter import py2ipynb

STATIC_FILES = Path('tests/static')


@pytest.mark.parametrize(
    "file_name",
    ['trivial_mapping_md_code_md', 'just_py_method', 'two_markdowns'],
)
def test_trivial_mapping_md_code_md(file_name):
    python_file_path, json_file_path = Path(file_name + '.py'), Path(file_name + '.json')

    file_py = (STATIC_FILES / python_file_path).read_text()
    transformed_py = py2ipynb(file_py)

    file_json = (STATIC_FILES / json_file_path).read_text()
    json_dict = json.loads(file_json)

    diff = DeepDiff(json_dict['cells'], transformed_py['cells'], ignore_type_in_groups=[(NotebookNode, dict)],
                    exclude_regex_paths=r"root\[\d+\]\['id'\]")

    assert diff == {}
