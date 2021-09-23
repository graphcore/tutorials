from enum import Enum


def supported_types():
    return list(OutputTypes)


class OutputTypes(str, Enum):
    JUPYTER_TYPE = "jupyter"
    MARKDOWN_TYPE = "markdown"
    PUREPYTHON_TYPE = "purepython"
