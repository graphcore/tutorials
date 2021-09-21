import os
from os.path import abspath, splitext
from pathlib import Path
from typing import List


def get_tests_dir():
    here, extension = splitext(Path(__file__).parent)
    return abspath(here)


def remove_files_if_present(files: List[Path]):
    for f in files:
        if os.path.exists(f):
            os.remove(f)
