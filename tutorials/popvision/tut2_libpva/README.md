Tutorial 2: Accessing profiling information
-----------------------------------------------------

For this tutorial we are going to use a PopART MNIST example and capture profile information that can be read using the PopVision Analysis Library, which is included in the Poplar SDK package.

The PopART MNIST example is in the `start_here` directory (this is a subset of the code from the [simple_applications/popart/mnist](../../../simple_applications/popart/mnist) directory). Enter the `start_here` directory and follow the instructions in the README.md to install the required modules and download the data.

Once you have followed the instructions and been able to run the MNIST example we will re-run the MNIST example with profiling enabled:

    POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true"}' python3 popart_mnist.py

When this has completed you will find a file called profile.pop in the training and inference subdirectories of the current working directory. Note: You can specify an output directory for the profile files to be written to:

    POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"mydirectory"}' python3 popart_mnist.py

Before we start using libpva, you should familiarise yourself with the documentation which can be found here: <a href=https://docs.graphcore.ai/projects/libpva/en/latest/api-python.html>PopVision Analysis Python API</a>.

Start Python in the directory that contains the profile.pop file you would like to read. Loading the profile into a Python object is easily done with the following:

    import pva
    report = pva.openReport('profile.pop')

Now you can access information from the report, as shown in the following examples:

    print("Number of compute sets: ", report.compilation.graph.numComputeSets)
    print("Number of tiles on target: ", report.compilation.target.numTiles)
    print("Version of Poplar used: ", report.poplarVersion.string)

Try executing these examples yourself and you should see output similar to the following:

    Number of compute sets:  29
    Number of tiles on target:  1472
    Version of Poplar used:  2.1.0 (df2b00ba5a)

You can also iterate over properties such as execution steps, which each represent the execution of a program in Poplar. In this example, we sum the total number of cycles on IPU 0 for all execution steps:

    sum(step.ipus[0].cycles for step in report.execution.steps)

To analyse the compiled program, it is best to use a ProgramVisitor class with the appropriate visitor functions. For example, the following class will print the name of any OnTileExecute programs that use multiple vertices:

    class TestVisitor(pva.ProgramVisitor):
        def visitOnTileExecute(self, onTileExecute):
            if len(onTileExecute.computeset.vertices) > 1:
                print(onTileExecute.name)

Now we will apply this visitor to every program so that we can see a list of all OnTileExecute programs executed that use multiple vertices:

    v = TestVisitor()
    for s in report.execution.steps:
        s.program.accept(v)

NOTE: You may see a long list of identical names. This is due to multiple OnTileExecute steps having the same name, which is to be expected.

You can easily create plots of information using Python's matplotlib library. The following example plots total memory usage (including gaps) for each tile.

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot([tile.memory.total.includingGaps for tile in report.compilation.tiles])
    plt.xlabel('Tiles')
    plt.ylabel('Bytes')
    plt.savefig('MemoryByTilePlot.png')

Now open the newly created MemoryByTilePlot.png file and you should see a plot similar to the following.

![PopVision Analysis Library screenshot of memory by tile](./screenshots/bytesByTile.png)

The examples shown in this tutorial are available in the complete/libpva_examples.py Python script, which you may run from any directory that contains a profile.pop file. Alternatively, perhaps you would like the challenge of finishing the incomplete version of this script in start_here/libpva_examples.py.

Copyright (c) 2021 Graphcore Ltd. All rights reserved.
