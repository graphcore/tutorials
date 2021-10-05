# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import re
from typing import Optional

from nbconvert import Exporter
from nbconvert.exporters.exporter import ResourcesDict
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell

from src.constants import REGEX_COPYRIGHT_PATTERN


class CodeExporter(Exporter):
    file_extension = '.py'

    def __init__(self, **kw):
        super().__init__(**kw)

    def from_notebook_node(self, notebook: NotebookNode, **kwargs):
        copyright_cell = self._find_first_copyright_node(notebook)
        code_cells = [copyright_cell.source] if copyright_cell else []

        code_cells = code_cells + [cell.source + os.linesep for cell in notebook.cells if cell.cell_type == 'code']

        py_code = os.linesep.join(code_cells)
        return py_code, ResourcesDict()

    @classmethod
    def _find_first_copyright_node(cls, notebook) -> Optional[NotebookNode]:
        """
        Simple search method with early exit to find the first copyright Markdown node, and transform it into
        a code cell. This way it will be part of this exporters output.
        """
        pattern = re.compile(pattern=REGEX_COPYRIGHT_PATTERN, flags=re.RegexFlag.IGNORECASE)
        for cell in notebook.cells:
            if cell.cell_type == 'markdown' and pattern.match(cell.source):
                code_cell = new_code_cell(source=cell.source)
                code_cell.source = os.linesep.join([f'# {line}' for line in code_cell.source.splitlines()])
                return code_cell
        return None
