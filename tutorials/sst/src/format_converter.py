import os
from enum import Enum
from typing import List

from nbformat import NotebookNode
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from src.constants import CELL_SEPARATOR


def code_preprocessor(input_source: str) -> str:
    return input_source.strip(os.linesep)


def markdown_preprocessor(input_source: str) -> str:
    return input_source


class CellType(Enum):
    CODE = 1
    MARKDOWN = 2


type2func = {
    CellType.CODE: new_code_cell,
    CellType.MARKDOWN: new_markdown_cell,
}
type2preprocessor = {
    CellType.CODE: code_preprocessor,
    CellType.MARKDOWN: markdown_preprocessor,
}


def py_to_ipynb(py_file_text: str) -> NotebookNode:
    cells = []

    current_cell_type = CellType.CODE
    cell_lines = []

    for line in py_file_text.splitlines():
        if not line.startswith(CELL_SEPARATOR):
            cell_lines.append(line)
            continue

        if cell_lines:
            new_cell = create_cell_from_lines(cell_lines=cell_lines, cell_type=current_cell_type)
            cells.append(new_cell)
            cell_lines = []

        if current_cell_type == CellType.CODE:
            current_cell_type = CellType.MARKDOWN
        elif current_cell_type == CellType.MARKDOWN:
            current_cell_type = CellType.CODE

    if cell_lines:
        new_cell = create_cell_from_lines(cell_lines, current_cell_type)
        cells.append(new_cell)

    notebook = new_notebook(cells=cells)

    return notebook


def create_cell_from_lines(cell_lines: List[str], cell_type: CellType) -> NotebookNode:
    source = os.linesep.join(cell_lines)
    processed_source = type2preprocessor[cell_type](source)
    cell = type2func[cell_type](processed_source)
    return cell
