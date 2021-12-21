# Copyright (c) 2020 Graphcore Ltd. All rights reserved.


def pytest_sessionstart(session):
    # We import after the start of the session to allow the tests
    # to be discovered without requiring test specific dependencies.
    from imdb import get_dataset
    print("Getting IMDB dataset...")
    get_dataset()
