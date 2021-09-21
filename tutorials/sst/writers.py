import os
from os import linesep
from os.path import splitext
from pathlib import Path
from typing import Optional

from nbformat import v4, NotebookNode


def write_nodes_to_json(notebook, output):
    jsonform = v4.writes(notebook)
    with open(output, "w") as fpout:
        fpout.write(jsonform)


def write_nodes_to_pure_python(notebook: NotebookNode, output):
    code_cells = [cell.source for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']

    py_code = linesep.join(code_cells + [os.linesep])
    with open(output, "w") as fpout:
        fpout.write(py_code)


def autogenerate_path_if_needed(filepath: Optional[Path], suffix: str, template: Path) -> Path:
    if filepath:
        return filepath

    template_name, extension = splitext(template)

    return template.parent / f"{template_name}{suffix}{extension}"
