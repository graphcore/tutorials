from nbconvert.preprocessors import ExecutePreprocessor
from tqdm import tqdm


class ExecutePreprocessorWithProgressBar(ExecutePreprocessor):
    def __init__(self, **kw):
        super().__init__(**kw)

    def preprocess(self, nb, resources=None, km=None):
        nb.cells = ProgressList(nb.cells, desc="Executing cells")
        nb, resources = super().preprocess(nb=nb, resources=resources, km=km)
        nb.cells = nb.cells.unwrap()
        return nb, resources


class ProgressList(list):
    def __init__(self, collection, desc, leave=False):
        super().__init__()
        self._collection = collection
        self.progress_wrapper = tqdm(self._collection, leave=leave, total=len(self._collection), desc=desc)
        self.progress_wrapper.disable = False

    def unwrap(self):
        return self._collection

    def __iter__(self):
        for elem in self.progress_wrapper:
            yield elem

    def __getitem__(self, index):
        return self._collection[index]

    def __delitem__(self, index):
        del self._collection[index]

    def __setitem__(self, index, val):
        self._collection[index] = val
