# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re

from nbconvert.preprocessors import RegexRemovePreprocessor
from traitlets import Enum
from traitlets.config import Config

from src.constants import SST_HIDE_OUTPUT_TAG, REGEX_COPYRIGHT_PATTERN


def configure_tag_removal_preprocessor(c: Config):
    c.TagRemovePreprocessor.remove_all_outputs_tags = (SST_HIDE_OUTPUT_TAG,)
    c.TagRemovePreprocessor.enabled = True
    return c


def configure_extract_outputs_preprocessor(c: Config):
    c.ExtractOutputsPreprocessor.enabled = True
    return c


def configure_copyright_regex_removal_preprocessor(c: Config):
    c.RegexWithFlagsRemovePreprocessor.patterns = [REGEX_COPYRIGHT_PATTERN]
    c.RegexWithFlagsRemovePreprocessor.flag = re.RegexFlag.IGNORECASE
    c.RegexWithFlagsRemovePreprocessor.enabled = True
    return c


class RegexWithFlagsRemovePreprocessor(RegexRemovePreprocessor):
    """
    Extending the nbconvert class because it does not support flags and because it compiles all patterns into one,
    normal annotators like (?i) will not work, because they have to be at the start of a pattern.
    """
    flag = Enum(values=re.RegexFlag, default_value=re.RegexFlag.UNICODE).tag(config=True)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.compiled_patterns = [re.compile(pattern, flags=self.flag) for pattern in self.patterns]

    def check_conditions(self, cell) -> bool:
        matches = filter(None, [compiled_pattern.match(cell.source) for compiled_pattern in self.compiled_patterns])
        return not list(matches)
