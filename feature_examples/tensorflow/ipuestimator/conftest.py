# Copyright (c) 2020 Graphcore Ltd. All rights reserved.


def pytest_sessionstart(session):
    """Load the cifar10 data at the start of the session"""
    # We import after the start of the session to allow the tests
    # to be discovered without requiring test specific dependencies.
    from tensorflow.keras.datasets import cifar10

    cifar10.load_data()
