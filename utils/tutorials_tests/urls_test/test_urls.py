# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Iterable, List, Optional, Set, Tuple
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings
import requests

from tutorials_tests.testing_util import get_file_list

LINK_PATTERN_MD = re.compile(r"]\(<?([^)>\s\*]+)[)>\s]")

# TODO Doesn't find links targeting within same page
LINK_PATTERN_RST = re.compile(r"[^`]`[^`]*<([^>]+)>`")

DISALLOWED_CHAR = r"\s\]\"\\>`')}:,"
LINK_PATTERN_RAW = re.compile(rf"(https?:[^{DISALLOWED_CHAR}]*[^{DISALLOWED_CHAR}\.])")

# Match HTTP and HTTPS URL rather than file link or page link
URL_PATTERN = re.compile(r"^https?:")

LINK_PATTERN_PAGE = re.compile(r"^#")

# URLs which don't exist yet (e.g documentation for a future release) can be
# added to the list of exceptions below.
#
# Make sure to add a TODO(TXXXX) comment to remove the exception once the link
# is fixed.
EXCEPTIONS: List[str] = []


def get_all_links_from_file(file_path: Path) -> Set[str]:
    """
    Takes: a file path (markdown, rst, Jupyter notebook)
    Returns: all links found in the file
    """
    print(f"Reading {file_path}")

    with open(file_path, "r", encoding="latin-1") as file:
        file_contents_lines = file.readlines()

    join_char = "" if file_path.suffix == ".rst" else " "
    file_contents = join_char.join([line.strip() for line in file_contents_lines])

    compiled_re = LINK_PATTERN_RST if file_path.suffix == ".rst" else LINK_PATTERN_MD

    all_links = set(compiled_re.findall(file_contents))
    raw_links = set(LINK_PATTERN_RAW.findall(file_contents))

    additional_raw = [
        link for link in raw_links.difference(all_links)
    ]
    if additional_raw:
        warnings.warn(f"Raw link(s) found in: {file_path} {additional_raw}")

    return all_links | raw_links


def check_url_works(url: str) -> Optional[Tuple[str, str, int]]:
    """
    Checks given `url` is responsive. If an error occurs return the error
    response string and code. Returns `None` if good.
    """
    try:
        response = requests.head(url)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Allow the test to succeed with intermittent issues.
        # (TooManyRedirects is not caught as could be a broken url.)
        print(f"\t{url} -> ConnectionError/Timeout")
        return None

    code = response.status_code
    message = requests.status_codes._codes[code][0]  # pylint: disable=protected-access

    print(f"\t{url} -> {message} ({code})")

    if response.status_code == 302:
        check_url_works(response.headers["Location"])
    else:
        # Allow any non 4xx status code, as other failures could be temporary
        # and break the CI tests.
        if response.status_code >= 400 and response.status_code < 500:
            return url, message, code

    return None


def check_file_links(file_path: Path, links: Iterable[str]) -> List[str]:
    """
    Checks given list of file links are all valid relative to given filename.
    Returns list of failed links.
    """
    failed_paths = []

    for link in links:
        if "mailto:support@graphcore.ai" in link:
            print(f"SKIPPING EMAIL: {link}")
            continue

        link_target = file_path.parent / link
        if link_target.exists():
            print(f"\t{link_target} -> EXISTS")
        else:
            print(f"\t{link_target} -> NON-EXISTANT")
            failed_paths.append(f"{file_path}: {link_target} NON-EXISTANT")

    return failed_paths


def test_all_links() -> None:
    """
    pytest to test links from markdown, RST and Notebooks are valid.
    """
    root_path = Path(__file__).parents[3]
    text_types = (".md", ".rst", ".ipynb")
    file_list = get_file_list(root_path, text_types)

    failed_urls = []

    executor = ThreadPoolExecutor()
    for file in file_list:
        links = get_all_links_from_file(file)
        external_links = {link for link in links if URL_PATTERN.match(link)}

        # TODO Test links within same page are good
        page_links = {link for link in links if LINK_PATTERN_PAGE.match(link)}

        for url_result in executor.map(check_url_works, external_links):
            if url_result is not None:
                url, message, code = url_result
                if url in EXCEPTIONS:
                    print(f"{url} found in exceptions: ignoring {message} ({code})")
                else:
                    failed_urls.append(f"{file}: {url} {message} ({code})")

        file_links = links.difference(external_links).difference(page_links)
        failed_urls += check_file_links(Path(file), file_links)

    no_failures = not failed_urls
    assert no_failures, "\n".join(failed_urls)
