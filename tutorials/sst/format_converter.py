from enum import Enum
from typing import Dict, List

from nbformat import NotebookNode, v4
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def code_preprocessor(input_source: str) -> str:
    return input_source.strip('\n')


def markdown_preprocessor(input_source: str) -> str:
    return input_source


class CellProcessingStatus(Enum):
    CODE = 1
    MARKDOWN = 2


class FormatConverter:
    type2func = {
        CellProcessingStatus.CODE: new_code_cell,
        CellProcessingStatus.MARKDOWN: new_markdown_cell,
    }
    type2preprocessor = {
        CellProcessingStatus.CODE: code_preprocessor,
        CellProcessingStatus.MARKDOWN: markdown_preprocessor,
    }

    def py2ipynb(self, py_file_text: str) -> NotebookNode:
        cells = self.extract_cells(py_file_text)
        notebook = new_notebook(cells=cells)
        return notebook

    def extract_cells(self, python_text_file) -> List[NotebookNode]:
        cells = []

        current_cell_type = CellProcessingStatus.CODE
        cell_lines = []

        for line in python_text_file.splitlines():
            if not line.startswith('"""'):
                cell_lines.append(line)
                continue

            if cell_lines:
                new_cell = self.get_cell(cell_lines, current_cell_type)
                cells.append(new_cell)
                cell_lines = []

            if current_cell_type == CellProcessingStatus.CODE:
                current_cell_type = CellProcessingStatus.MARKDOWN
            elif current_cell_type == CellProcessingStatus.MARKDOWN:
                current_cell_type = CellProcessingStatus.CODE

        if cell_lines:
            new_cell = self.get_cell(cell_lines, current_cell_type)
            cells.append(new_cell)

        return cells

    def get_cell(self, cell_lines: List[str], current_cell_type: CellProcessingStatus) -> NotebookNode:
        source = '\n'.join(cell_lines)
        processed_source = self.type2preprocessor[current_cell_type](source)
        cell = self.type2func[current_cell_type](processed_source)
        return cell
