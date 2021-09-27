import os
from os.path import abspath, splitext
from pathlib import Path


def get_tests_dir():
    here, extension = splitext(Path(__file__).parent.parent.parent / 'tests')
    return abspath(here)


STATIC_FILES = Path(get_tests_dir() + os.sep + 'static')

EXAMPLE_INPUT_PATH = STATIC_FILES / "trivial_mapping_md_code_md.py"
