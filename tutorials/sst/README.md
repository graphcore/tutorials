# Single Source Tool

Documentation maintenance tool for converting python files to jupyter notebooks and markdown.

## Installation
```bash
cd sst
pip3 install -e .
pytest .
```

Run script by typing:
```bash
sst --help
```
or:
```bash
python3 sst.py --help
```
## Design assumptions

This section describes how the python file is processed.

Text included in a long comment is treated as a single markdown cell. Python code between comments is treated as 
a single cell. For example:
```python
"""
This is the first cell, in it you can use all the things that are available in typical markdown:
**bold**, *italic*, code, url, images, headers, lists etc.
"""

print('Hello world')

"""
This is a second cell
"""
```

You can use an empty comment to split the python code into two separate cells:

```python
def hello(name):
    print(f'Hello {name}!')
    
""""""

hello('John')
```