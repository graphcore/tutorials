from pathlib import Path
from typing import Tuple

from src.output_types import OutputTypes, EXTENSION2TYPE, TYPE2EXTENSION, supported_types


def set_output_extension_and_type(output: Path, type: OutputTypes) -> Tuple[Path, OutputTypes]:
    """
    Handles input given by the user. The user can define the type to which the file will be converted directly using
    the switch or using the extension in the path to the output. In the result of the function the type variable is set
    correctly and the file has the specified extension. Additionally, the function validates the input given by the user.

    If output without extension but specified type -> add extension to output
    If output with extension -> overwrite current type
    If output with extension but not allowed extension -> raise AssertionError
    If output without extension and type is None -> raise AttributeError
    """
    if output.suffix != '':
        allowed_extensions = list(EXTENSION2TYPE.keys())
        assert output.suffix in allowed_extensions, \
            f'Specified output file has type: {output.suffix}, while only {allowed_extensions} are allowed.'
        type = EXTENSION2TYPE[output.suffix]
    elif type is not None:
        output = output.with_suffix(TYPE2EXTENSION[type])
    else:
        raise AttributeError(
            f'Please provide output file type by adding extension to outfile (.md or .ipynb) or specifying that by '
            f'--type parameter {supported_types()} are allowed.'
        )

    return output, type
