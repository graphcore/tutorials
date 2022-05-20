# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
from test_urls import LINK_PATTERN_MD, LINK_PATTERN_RAW


def test_md_link_with_alt_text() -> None:
    result = LINK_PATTERN_MD.search(
        'blah [Alt Text](figures/ExampleScreen.png "TensorBoard Example") blah'
    )
    assert result, "Regex failed to find a match"
    assert result.group(1) == "figures/ExampleScreen.png"


def test_md_link_with_extra_angle_brackets() -> None:
    result = LINK_PATTERN_MD.search(
        "blah [Alt Text](<https://docs.graphcore.ai/>) blah"
    )
    assert result, "Regex failed to find a match"
    assert result.group(1) == "https://docs.graphcore.ai/"


def test_md_should_not_match_python_list_callable() -> None:
    result = LINK_PATTERN_MD.search('args = layer["func"](*args)')
    print(result)
    assert not result, "Regex should ignore (*...)"


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("blah https://docs.graphcore.ai/ blah", "https://docs.graphcore.ai/"),
        ("blah `https://docs.graphcore.ai/` blah", "https://docs.graphcore.ai/"),
        ("blah 'https://docs.graphcore.ai/' blah", "https://docs.graphcore.ai/"),
        ("blah <https://docs.graphcore.ai/> blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/. blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/, blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/: blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/} blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/] blah", "https://docs.graphcore.ai/"),
        (r"blah https://docs.graphcore.ai/\n blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/\n blah", "https://docs.graphcore.ai/"),
        ('blah https://docs.graphcore.ai/" blah', "https://docs.graphcore.ai/"),
    ],
)
def test_raw_http_single_link(test_input: str, expected: str) -> None:

    result = LINK_PATTERN_RAW.findall(test_input)
    assert (
        len(result) == 1
    ), f"Regex failed to match exactly one item. Found {len(result)}"
    assert result[0] == expected
