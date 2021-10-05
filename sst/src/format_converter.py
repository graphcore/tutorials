# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
from enum import Enum
from typing import List, Optional

from nbformat import NotebookNode
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from src.constants import CELL_SEPARATOR, SST_HIDE_OUTPUT_TAG, SHEBANG_MARKER


class CellType(Enum):
    CODE = 1
    MARKDOWN = 2


TYPE2FUNC = {
    CellType.CODE: new_code_cell,
    CellType.MARKDOWN: new_markdown_cell,
}


def py_to_ipynb(py_file_text: str) -> NotebookNode:
    """
    The python file content is parsed line by line. The type of fragment (markdown, codecell) is determined together
    with tags. Based on that data, an object which represents Notebook is created.

    Args:
        py_file_text: The content of a python file stored in a string variable

    Returns:
        NotebookNode file that represents Notebook object
    """
    cells = []

    current_cell_type = CellType.CODE
    cell_lines, cell_tags = [], []

    for line in py_file_text.splitlines():

        if SST_HIDE_OUTPUT_TAG in line and SST_HIDE_OUTPUT_TAG not in cell_tags:
            cell_tags.append(SST_HIDE_OUTPUT_TAG)

        if is_code_or_markdown(line):
            cell_lines.append(line)
        elif line.startswith(CELL_SEPARATOR):
            if cell_lines:
                new_cell = create_cell_from_lines(cell_lines=cell_lines, cell_type=current_cell_type,
                                                  cell_tags=cell_tags)
                if new_cell:
                    cells.append(new_cell)

                cell_lines, cell_tags = [], []

            if current_cell_type == CellType.CODE:
                current_cell_type = CellType.MARKDOWN
            elif current_cell_type == CellType.MARKDOWN:
                current_cell_type = CellType.CODE

    if cell_lines:
        new_cell = create_cell_from_lines(cell_lines=cell_lines, cell_type=current_cell_type, cell_tags=cell_tags)
        cells.append(new_cell)

    notebook = new_notebook(cells=cells)

    return notebook


def create_cell_from_lines(cell_lines: List[str], cell_type: CellType, cell_tags: List[str]) -> Optional[NotebookNode]:
    """
    Args:
        cell_lines: List of content in each line
        cell_type: Recognized cell type
        cell_tags: List of cell tags to add in metadata

    Returns:
        NotebookNode which contains specified type and merged content
    """
    source = os.linesep.join(cell_lines)
    processed_source = source.strip(os.linesep)

    cell = None
    if processed_source:
        cell = TYPE2FUNC[cell_type](processed_source)

        if cell_tags:
            cell.metadata.update({"tags": cell_tags})

    return cell


def is_code_or_markdown(line: str) -> bool:
    """
    Decide if the line should be in the result

    Args:
        line: string containing line content

    Returns:
        bool, True if it does not contain some special tag
    """
    return not line.startswith(CELL_SEPARATOR) and \
           not line.startswith(SHEBANG_MARKER) and \
           not SST_HIDE_OUTPUT_TAG in line
