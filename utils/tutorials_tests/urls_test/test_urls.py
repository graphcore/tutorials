# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import re
import requests

URL_PATTERN = re.compile(r"https?:[^\s>\)]+")
# Matches HTTP and HTTPS urls in Markdown and RST formats,
# up until a whitespace, > or ) characters.

# Some characters are rarely present at the end of a URL.
# If we find them, they're probably not meant to be part of the URL.
# We want to make sure they're removed from the URL before it's tested.
BAD_ENDINGS = ('.', ',', ']', ':', ';', '}', '\\n', '"')

# URLs which don't exist yet (e.g documentation for a future release) can be
# added to the list of exceptions below.
#
# Make sure to add a TODO(TXXXX) comment to remove the exception once the link
# is fixed.
EXCEPTIONS = []


def get_all_links_from_file(file_path):
    """
    Takes: a file path (markdown, rst, Jupyter notebook)
    Returns: all http/s links found in the file

    ! This assumes NO raw links in rst files, they will
    not be correctly returned.
    """
    print(f"Reading {file_path}")

    all_links = []

    # Force as extended ASCII to avoid decoding erors:
    # assume all urls are made of 8-bit chars only
    with open(file_path, 'r', encoding="latin-1") as file:
        # RST files can have links over several lines,
        # they need to be joined first so full URLs are found
        if file_path.endswith('.rst'):
            file = "".join([line.strip() for line in file.readlines()])
            matches = URL_PATTERN.findall(file)
            for match in matches:
                while match.endswith(BAD_ENDINGS):
                    # Remove character not meant to be in the URL
                    if match.endswith("\\n"):
                        match = match[:-2]
                    else:
                        match = match[:-1]
                all_links.append(match)
        else:
            for line in file:
                matches = URL_PATTERN.findall(line)
                for match in matches:
                    while match.endswith(BAD_ENDINGS):
                        # Remove character not meant to be in the URL
                        if match.endswith("\\n"):
                            match = match[:-2]
                        else:
                            match = match[:-1]
                    all_links.append(match)

    return all_links


def check_url_works(url):
    print(f"Testing {url}")

    try:
        r = requests.head(url)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Allow the test to succeed with intermitent issues.
        # (TooManyReditects is not caught as could be a broken url.)
        return

    code = r.status_code
    message = requests.status_codes._codes[code][0]  # pylint: disable=protected-access

    print(message + f" ({code})")

    if r.status_code == 302:
        check_url_works(r.headers['Location'])
    else:
        # Allow any non 4xx status code, as other failures could be temporary
        # and break the CI tests.
        if r.status_code >= 400 and r.status_code < 500:
            return url, message, code
        print()


def test_all_links():
    cwd = os.path.abspath(__file__)
    root_path = os.path.abspath(os.path.join(cwd, "..", "..", "..", ".."))

    text_types = ('.md', '.rst', '.ipynb')
    failed_urls = []

    for path, _, files in os.walk(root_path):
        # Select only text files
        text_files = [os.path.join(path, file_name)
                      for file_name in files if file_name.endswith(text_types)]

        for file in text_files:
            for url in get_all_links_from_file(file):
                url_result = check_url_works(url)
                if url_result is not None:
                    url, message, code = url_result
                    if url in EXCEPTIONS:
                        print(f"{url} found in exceptions: ignoring {message} ({code})")
                    else:
                        failed_urls.append(f"{url}: {message} ({code})")
                print()

    assert not failed_urls, failed_urls
