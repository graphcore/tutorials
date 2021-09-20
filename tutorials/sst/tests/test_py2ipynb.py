import json
from pathlib import Path

from deepdiff import DeepDiff

from format_converter import FormatConverter

STATIC_FILES = Path('tests/static')


def test_trivial_mapping_md_code_md():
    # TODO parametrized test
    file_name = 'trivial_mapping_md_code_md'
    python_file_path, json_file_path = Path(file_name + '.py'), Path(file_name + '.json')

    file_py = (STATIC_FILES / python_file_path).read_text()
    file_json = (STATIC_FILES / json_file_path).read_text()
    json_dict = json.loads(file_json)

    transformed_py = FormatConverter().py2ipynb(file_py)
    diff = DeepDiff(json_dict, transformed_py)

    assert diff == {}
