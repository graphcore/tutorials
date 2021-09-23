from enum import Enum


def supported_types():
    return list(OutputTypes)


class OutputTypes(str, Enum):
    JUPYTER_TYPE = "jupyter"
    MARKDOWN_TYPE = "markdown"
    PUREPYTHON_TYPE = "purepython"


EXTENSION2TYPE = {
    '.ipynb': OutputTypes.JUPYTER_TYPE,
    '.md': OutputTypes.MARKDOWN_TYPE,
    '.py': OutputTypes.PUREPYTHON_TYPE
}
TYPE2EXTENSION = {type: extension for extension, type in EXTENSION2TYPE.items()}
