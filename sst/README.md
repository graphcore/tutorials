# Single Source Tool

Documentation maintenance tool for converting python files to jupyter notebooks and markdown documents. 

The purpose of this software is to keep the documentation in one format (a python file with a established 
documentation convention) when the script is able to run it and transform it to other formats (markdown document or 
jupyter notebook), or hide the documentation completely and create a ready to run script.

## Installation

In order to install the package together with its dependencies, run:
```bash
pip3 install -e .
```

This will allow you to run the script from any directory by typing:
```bash
sst --help
```

However, you can still install the dependencies separately or use the package by calling the script:
```bash
pip install -r requirements.txt
python3 sst.py --help
```

## Python file convention
    
This section describes what the syntax of a python file is.


Text placed in triple quotes without indentation is treated as markdown cell. So if you would like to create 
a documentation cell describing your code, you can use the example syntax: 
```python
"""
This is the first cell, here you are able to use whole syntax that is available in typical markdown. For example
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
Rest of installation
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
will take up a lot of space. You can hide such output from such a cell by adding a special comment to your code: 
`# sst_hide_output` - note that line with this tag will be hidden. For example:

```python
for _ in range(1000):
    print(greet('John'))
# sst_hide_output
```

#### Special handling
This tool implements special handling for few particular cases of words or technical elements that will be handled in 
a characteristic way.

>**Shebangs**
> 
>When a script contains lines, which start with `#!`, the whole line will be removed from all outputs.

>**Copyright notice**
> 
> Each markdown cell with word `copyright` (case-insensitive) will be removed from Markdown file, while in other types it will remain.


## Transformation to other formats
By default, beyond python file that is single source of truth we would like to store in the repository:
- jupyter notebook which was not executed 
- markdown file with the outputs after the execution
- python code file without documentation

You can automatically generate all 3 formats by using the command:
```bash
sst convert2all \
 --source path_to_your_python_file \
 --output-dir path_to_your_directory \
 --markdown-name XYZ            # [Optional with default 'README.md']
```
The resulting files will have the same names as the input file with a different extension, with two exceptions:
- the Markdown output, which is either 'README.md' or custom provided name using the optional argument `markdown-name.`
- the Python output code, in which case to avoid overwriting the source file the suffix `_code_only` will be added to it. Example result will be placed under:
```bash
path_to_your_directory/XYZ.md
path_to_your_directory/source_file_name.ipynb
path_to_your_directory/source_file_name_code_only.py
```

If you would like to create each of these format files separately use:
```bash
sst convert \
 --source path_to_you_python_file \
 --output outputfile \
 --type [jupyter|markdown|code] \
 --execute/--no-execute
```

Switch `--execute/--no-execute` indicates whether the file should be run. This implies the resulting file will contain 
the outputs from the run or only the contents of the source file itself.

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
sst batch-convert \
 --config configuration_file.yml \
 --source-dir directory_where_script_will_be_executed \
 --output-dir directory_where_outputs_will_be_stored \
 --execute
 ```

Config file would look like this:
```yaml
files:
    - name: file1
      source: pytorch/package/script.py
```

You can find more details by adding a `--help` switch to each command.
