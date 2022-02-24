# PopART Recomputing example

This example runs generated data through a seven layer DNN.
It shows how to use manual and automatic recomputation in popART.

Manual recomputing allows the user to choose which ops to recompute.

Automatic recomputing uses an heuristic technique.
See https://arxiv.org/abs/1604.06174


## How to run the example

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the [Getting Started guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU system.

2) Install the required packages.

```cmd
python3 -m pip install -r requirements.txt
```

3) Run the program. Note that the PopART Python API only supports Python 3.

```cmd
python3 recomputing.py
```

### Options

The program has a few command-line options:

`--help` Show usage information

`--export FILE` Export the model created to FILE

`--recomputing STATUS` Choice amongst ON (default), AUTO and OFF

* ON recomputes activations for all but checkpointed layers
* AUTO uses popART auto recomputation
* OFF deactivate recomputing altogether

`--show-logs` show execution logs

### Profiling

The Poplar SDK can generate report files during the compilation and execution of applications.
The following enables report generation, and specifies a directory to generate reports in.

```
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"."}'
```

After running an application, the resulting report files can be opened using the PopVision Graph Analyser.
See the PopVision user guide for more information:
[PopVision User Guide](https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/index.html).

## Run the tests

Install the required packages, then use pytest to run the example in all three recomputation modes.

```cmd
python3 -m pip install -r requirements.txt
python3 -m pytest
```
