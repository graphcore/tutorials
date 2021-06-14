# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import subprocess
from pathlib import Path
from filelock import FileLock


def pytest_sessionstart(session):
    build_dir = Path(__file__).parent.resolve()
    with FileLock(build_dir.joinpath("binary.lock")):
        subprocess.run("make clean", cwd=build_dir, shell=True, check=True)
        subprocess.run("make", cwd=build_dir, shell=True, check=True)
