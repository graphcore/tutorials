from nbconvert import NotebookExporter, MarkdownExporter

from src.python_exporter import PythonExporter

EXECUTE_PREPROCESSOR = 'nbconvert.preprocessors.ExecutePreprocessor'
CELL_SEPARATOR = '"""'
TYPE2EXPORTER = {
    'jupyter': NotebookExporter,
    'markdown': MarkdownExporter,
    'purepython': PythonExporter
}