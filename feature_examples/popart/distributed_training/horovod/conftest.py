# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
from pathlib import Path
import sys
# Add path for mnist utilities to reuse code from common.py
mnist_path = Path(Path(__file__).absolute().parent.parent.parent,
                  'mnist')
sys.path.append(str(mnist_path))
from common import download_mnist


def pytest_sessionstart(session):
    """Download the mnist data at the start of the session"""
    download_mnist(os.path.dirname(__file__))
