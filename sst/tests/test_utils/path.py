# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from os.path import abspath, splitext
from pathlib import Path


def get_tests_dir() -> str:
    """
    This utility function always returns the correct path to the 'tests' directory of SST application.
    It is implemented by a path relation between this file and it's parent directories.
    This is useful for CI/CD tools which can start the tests from various starting point directories - this function
    return the absolute path which will work on all platforms.
    """
    here, extension = splitext(Path(__file__).parent.parent.parent / 'tests')
    return abspath(here)


def get_unit_test_static_files_dir() -> Path:
    """
    This utility function always returns the location of all the static files used by unit-tests.
    """
    return Path(get_tests_dir()) / 'static'
