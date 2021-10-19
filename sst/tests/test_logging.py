import pytest

from src.execute import warning_output_location_differs_with_source
from src.utils.logger import get_logger
from tests.test_utils.logger_utils import enable_logging, MockHandler


@pytest.fixture
def setup():
    enable_logging()
    logger_instance = get_logger()
    stream = []
    logger_instance.addHandler(MockHandler(stream))
    return logger_instance, stream


def test_logger(setup):
    logger_instance, stream = setup
    logger_instance.warn("hello")

    assert stream == ["hello"]


def test_output_location_warning(setup, tmp_path):
    logger_instance, stream = setup

    output_dir = tmp_path / 'output_dir' / 'test.ipynb'
    input_dir = tmp_path / 'input_dir' / 'test.py'

    warning_output_location_differs_with_source(
        output=output_dir,
        source=input_dir
    )

    assert stream
    assert "Outputs will be generated in a location different" in stream[0]


def test_output_location_warning_same_locations(setup, tmp_path):
    logger_instance, stream = setup

    output_dir = tmp_path / 'input_dir' / 'test.ipynb'
    input_dir = tmp_path / 'input_dir' / 'test.py'

    warning_output_location_differs_with_source(
        output=output_dir,
        source=input_dir
    )

    assert not stream
