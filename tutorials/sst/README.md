# Single Source Tool

Documentation maintenance tool for converting python files to jupyter notebooks and markdown. 

The purpose of this software is to keep the documentation in one format (a python file with a established 
documentation convention) when the script is able to run it and transform it to other formats (markdown or 
jupyter notebook), or hide the documentation completely and create a ready to run script (later called pure python).

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

## Python file convention
    
This section describes what the syntax of a python file is.


Text placed in triple quotes without indentation is treated as markdown cell. So if you would like to create 
a documentation cell describing your code, you can use the example syntax: 
```python
"""
This is the first cell,here you are able to use whole syntax that is available in typical markdown. For example
**bold**, *italic*, `code`, url, images, headers, lists etc.
"""
```

Everything except triple quotes is treated as code cell, however you can add multiple markdown cells after each other:
```python
"""
# Motivation
Our software is designed to solve all your problems
"""
"""
# Installation
Rest of tutorial
"""
"""
Example usage:
"""

def greet(name: str):
    print(f'Hello {name}')

greet('John')

"""
# Contribution
....
"""
```

#### Hiding outputs
Sometimes it may happen that your post-execution file contains very long output, which when transformed to markdown 
will take up a lot of space. You can hide output from such a cell by adding a special comment to your code: 
`# remove_output`. For example:

```python
for _ in range(1000):
    print(greet('John'))
# remove_output
```

## Transformation to other formats
By default, we expect in the repository store beyond python file that is single source of truth :
- jupyter notebook which was not executed 
- markdown file with the outputs after the execution
- pure python file without documentation

You can automatically generate all 3 formats by using the command:
```bash
sst convert2all --source path_to_you_python_file --output-dir path_to_your_directory
```

If you would like to create each of these format files separately use:
```bash
sst convert \
 --source path_to_you_python_file \
 --output outputfile \
 --type [jupyter|markdown|purepython] \
 --execute/--no-execute
```

Example usages:
```bash
sst convert \
 --source myfile.py \
 --output outputfile_path \
 --type jupyter \
 --execute
 ```

You can also define the output type by specifying the appropriate extension in output:
```bash
sst convert \
 --source myfile.py \
 --output outputfile.md \
 --no-execute
 ```

Moreover, you can convert a whole batch of python scripts, using a yaml configuration file:

```bash
sst batch_convert \
 --config tutorial_list.yml \
 --source-dir tutorials/tutorials \
 --output-dir tutorials/tutorials/my_batch \
 --execute
 ```

Config file would look like this:
```yaml
tutorials:
    - name: Tutorial1
      source: pytorch/tutorial1/script.py
```

You can find more details by adding a `--help` switch to each command.