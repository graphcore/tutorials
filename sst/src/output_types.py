# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from enum import Enum


def supported_types():
    return list(OutputTypes)


class OutputTypes(str, Enum):
    JUPYTER_TYPE = "jupyter"
    MARKDOWN_TYPE = "markdown"
    CODE_TYPE = "code"

    def __repr__(self):
        return self.value


EXTENSION2TYPE = {
    '.ipynb': OutputTypes.JUPYTER_TYPE,
    '.md': OutputTypes.MARKDOWN_TYPE,
    '.py': OutputTypes.CODE_TYPE
}
TYPE2EXTENSION = {type: extension for extension, type in EXTENSION2TYPE.items()}
