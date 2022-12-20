""""
Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
"""
# Tutorial: Reading PVTI files with libpva

When using the
[PopVision System Analyser](https://www.graphcore.ai/developer/popvision-tools)
to examine profiling information from an IPU application's host machine, you
may find yourself with specific analysis needs that the GUI doesn't provide.
This tutorial covers functionality from the PopVision Analysis Library
(`libpva`) that allows you to read the PopVision Trace Information (PVTI) files
that are read by the System Analyser, allowing you to perform your own
filtering and analysis. In this tutorial you will:

- Trace program events when running an application and create a PVTI file
- Programmatically load PVTI files
- Read and traverse the PVTI data model
- Perform some basic statistical analysis on a specific subset of events

If this is your first time using the PopVision Analysis tools you may prefer
to use the PopVision System Analyser as described in
[its user guide](https://docs.graphcore.ai/projects/system-analyser-userguide/en/2.11.2/)
to familiarise yourself with some of the information available.

"""
"""
## How to run this tutorial
"""
"""
To run the Python version of this tutorial:

1. Download and install the Poplar SDK. Run the `enable.sh` scripts for Poplar and PopART as described in the [Getting
  Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide for your IPU system.
2. For repeatability we recommend that you create and activate a Python virtual environment. You can do this with:
   a. create a virtual environment in the directory `venv`: `virtualenv -p python3 venv`;
   b. activate it: `source venv/bin/activate`.
3. Install the Python packages that this tutorial needs with `python -m pip install -r requirements.txt`.
4. Download the MNIST dataset we're using with `./get_data.sh`. You should now have a 'data' folder containing the MNIST dataset

sst_ignore_jupyter
"""
"""
To run the Jupyter notebook version of this tutorial:

1. Enable a Poplar SDK environment (see the [Getting
  Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide for
  your IPU system)
2. In the same environment, install the Jupyter notebook server:
   `python -m pip install jupyter`
3. Launch a Jupyter Server on a specific port:
   `jupyter-notebook --no-browser --port <port number>`
4. Connect via SSH to your remote machine, forwarding your chosen port:
   `ssh -NL <port number>:localhost:<port number>
   <your username>@<remote machine>`

For more details about this process, or if you need troubleshooting, see our
[guide on using IPUs from Jupyter
notebooks](../../standard_tools/using_jupyter/README.md).
"""
"""
## Enabling PVTI file generation

Enable PVTI logging by setting the PVTI_OPTIONS environment variable and run
'popart_mnist.py' as follows:

```bash
PVTI_OPTIONS='{"enable":"true"}' python3 popart_mnist.py
```

When this has completed you will find a PVTI file in the working directory,
e.g. "Tue_Nov_24_11:59:17_2020_GMT_4532.pvti".

> **Note**: You can specify an output directory for the PVTI files to be
> written to:
>
> ```bash
> PVTI_OPTIONS='{"enable":"true", "directory": "tommyFlowers"}' python3 popart_mnist.py
> ```
"""
# %pip install -r requirements.txt
# ! sh ./get_data.sh
# ! PVTI_OPTIONS='{"enable":"true"}' python3 popart_mnist.py
# sst_ignore_md
# sst_ignore_code_only
"""
"""
# sst_ignore_md
# sst_ignore_jupyter
import os
import subprocess

os.environ["PVTI_OPTIONS"] = '{"enable":"true"}'
subprocess.run(["sh", "./get_data.sh"])
subprocess.run(["python3", "popart_mnist.py"])
"""
## Using the Python API

In this tutorial we use `libpva` to access PVTI profiling information. Refer to
the [libpva Python API documentation](https://docs.graphcore.ai/projects/libpva/en/3.1.0/api-python.html)
for more information.

"""
"""
### Loading a PVTI file

Start Python in the directory that contains the PVTI file you want to read.
Loading the file can be easily done with the following:
"""
# Grab the most recently modified PVTI file
from pathlib import Path

working_dir = Path(".").glob("./*.pvti")
pvti_files = [f for f in working_dir if f.is_file()]
pvti_files.sort(reverse=True, key=lambda a: a.stat().st_mtime)
trace_path = str(pvti_files[0])

# Open the file
import pva

trace = pva.openTrace(trace_path)
"""
### Accessing processes, threads, and events

Now you can access information from the report, such as the processes that were
running, their threads, the events on those threads, and the children of each
event:
"""
print("Number of processes: ", len(trace.processes))

process = trace.processes[0]
print("Number of threads on process ", process.pid, ": ", len(process.threads))

thread = process.threads[0]
print("Number of events on thread ", thread.tid, ": ", len(thread.events))

event = thread.events[0]
print("Event '", event.label, "' lasted ", event.duration, " microseconds")
print("The event has ", len(event.children()), " children")
"""
In this example we are retrieving processes and threads by their index in the
lists we get back, but if you already know the PIDs and TIDs of the threads you
want to explore (perhaps because you have seen them in the
[PopVision System Analyser](https://docs.graphcore.ai/projects/system-analyser-userguide/en/2.11.2/))
you can use `trace.process(pid)` and `process.thread(tid)` instead.

Events only have a few fields: a list of their children, a timestamp at which
they occurred, a duration, a label, and a channel. The label and channel are
determined by the instrumentation code within a host process. You can read more
about adding such instrumentation in the
[System Analyser Instrumentation](../system_analyser_instrumentation) tutorial.
"""
"""
### Analysing epochs

Now that we know how to use the API, let's use it to determine some useful
statistics. We'll be looking through our events and recording information about
our epoch blocks, and at the end we'll output the maximum, minimum, and
average durations.

To get started, let's prepare to record some statistics:
"""


class EpochStats:
    longest = None
    shortest = None
    average = 0
    count = 0


"""
Our events are structured as a tree, so we will traverse them with a recursive
function. We also only care about epoch blocks, so we focus on labels that
start with "Epoch":
"""


from typing import List


def traverse_events(events: List[pva.Event], stats: EpochStats):
    for event in events:
        # Record info about Epoch events
        if event.label.startswith("Epoch"):
            # Increase our count, and iteratively calculate the mean duration
            stats.count += 1
            stats.average += (event.duration - stats.average) / stats.count

            # Fill our slot for the longest event if the current event is
            # longer, or we just don't have one yet
            if stats.longest is None or event.duration > stats.longest.duration:
                stats.longest = event

            # Fill our slot for the shortest event if the current event is
            # shorter, or we just don't have one yet
            if stats.shortest is None or event.duration < stats.shortest.duration:
                stats.shortest = event

        # Traverse children too
        traverse_events(event.children(), stats)


"""
Now we open our trace and iterate through our processes and threads, passing
their events to our traversal function:
"""
trace = pva.openTrace(trace_path)
stats = EpochStats()

for process in trace.processes:
    for thread in process.threads:
        traverse_events(thread.events, stats)
"""
Finally, we can print our findings:
"""
print(
    f"The longest epoch '{stats.longest.label}' lasted {stats.longest.duration} microseconds."
)
print(
    f"The shortest epoch '{stats.shortest.label}' lasted {stats.shortest.duration} microseconds."
)
print(
    f"Epochs took {stats.average:.0f} microseconds on average, out of {stats.count} epochs in total."
)
"""
## Going further

In this tutorial, we wrote a simple script illustrating how to read data from a
PVTI file for a programmatic use-case. For general, day-to-day perusal of PVTI
files you will instead want to use the
[PopVision System Analyser](https://docs.graphcore.ai/projects/system-analyser-userguide/en/2.11.2/)
for a feature-rich, navigable graphical view over the trace data.

`libpva` also has a C++ API that is very similar to the Python API. For more
information about the C++ API, refer to the
[PopVision Analysis C++ API Documentation](https://docs.graphcore.ai/projects/libpva/en/3.1.0/api-cpp.html).

"""
