<!-- Copyright (c) 2022 Graphcore Ltd. All rights reserved. -->
# Data parallelism

This tutorial on Data parallelism in PopXL is available as a jupyter notebook.

To run the [notebook](mnist.ipynb) in this folder:

1. Install a Poplar SDK (version 2.6 or later) and source the enable.sh scripts
   for both PopART and Poplar as described in the [Getting Started
   guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU
   system.
2. Create a Python virtual environment: `python3 -m venv <virtual_env>`.
3. Activate the virtual environment: `. <virtual_env>/bin/activate`.
4. Update `pip`: `pip3 install --upgrade pip`
5. Install requirements `pip3 install -r requirements.txt` (this will also
   install `popxl.addons`).
6. Launch a Jupyter Server on a specific port:
   `jupyter-notebook --no-browser --port <port number>`. Be sure to be in
   the virtual environment.
7. Connect via SSH to your remote machine, forwarding your chosen port:
   `ssh -NL <port number>:localhost:<port number> <your username>@<remote machine>`

On the machine connected to IPUs:

```bash
python3 -m venv virtual_env
. virtual_env/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install jupyter
jupyter-notebook --no-browser --port 12345
```

Take note of the URL displayed by the `jupyter-notebook` command.

On your local machine:

```bash
ssh -NL 12345:localhost:12345 <your username>@<remote machine>
```

Then navigate in your web-browser to the URL displayed by Jupyter in the previous step.

For more details about this process, or if you need troubleshooting, see our
[guide on using IPUs from Jupyter
notebooks](../../standard_tools/using_jupyter/README.md).
