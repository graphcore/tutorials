# PopART Pipelining example

This demo shows how to use pipelining in PopART on a very simple model
consisting of two dense layers. Run one pipeline length and compute loss.

## File structure

* `pipelining.py` The main PopART file showcasing pipelining.
* `test_pipelining.py` Test script.
* `README.md` This file.

## How to use this example

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the [Getting Started guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU system.

2) Install the required packages.

```cmd
python3 -m pip install -r requirements.txt
```

3) Run the program. Note that the PopART Python API only supports Python 3.

```cmd
python3 pipelining.py [-h] [--export FILE] [--no_pipelining] [--test]
```

### Options

To list all the command line options, run:

```cmd
python3 pipelining.py -h
```

### Profiling

The Poplar SDK can generate report files during the compilation and execution of applications.
The following enables report generation, and specifies a directory to generate reports in.

```
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"."}'
```

After running an application, the resulting report files can be opened using the PopVision Graph Analyser.
See the Graph Analyser user guide for more information:
[PopVision Graph Analyser User Guide](https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/).

## Running the tests

Install the required packages and use pytest.

```cmd
python3 -m pip install -r requirements.txt
python3 -m pytest
```
