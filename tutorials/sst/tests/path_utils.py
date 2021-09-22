from os.path import abspath, splitext
from pathlib import Path


def get_tests_dir():
    here, extension = splitext(Path(__file__).parent)
    return abspath(here)
